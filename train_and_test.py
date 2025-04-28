import random
import time
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
import numpy as np
import torch
import wandb
from settings import num_classes


def cluster_sep_loss_fn(model, min_distances, label):

    max_dist = (model.module.prototype_shape[1]
                * model.module.prototype_shape[2]
                * model.module.prototype_shape[3]) ** 2
    batch_size = label.shape[0]
    cluster_cost = 0.0
    separation_cost = 0.0
    for b in range(batch_size):
        real_labels = torch.where(label[b] == 1)[0]
        multiple_cluster = []
        for one_label in real_labels:
            prototypes_of_correct_class = torch.t(model.module.prototype_class_identity[:, one_label]).cuda()
            inverted_distances = torch.max((max_dist - min_distances[b]) * prototypes_of_correct_class)
            multiple_cluster.append(max_dist - inverted_distances)
        cluster_cost += sum(multiple_cluster)
        # cluster_cost += sum(multiple_cluster) / len(multiple_cluster)

        prototypes_of_wrong_class = 1 - torch.t(model.module.prototype_class_identity[:, real_labels]).cuda()
        prototypes_of_wrong_class = prototypes_of_wrong_class.all(dim=0) * 1.0
        inverted_distances_to_nontarget_prototypes = torch.max((max_dist - min_distances[b]) * prototypes_of_wrong_class)
        separation_cost += max_dist - inverted_distances_to_nontarget_prototypes

    return cluster_cost / batch_size, separation_cost / batch_size


def distillation_single(student, teacher, temperature=2.0):  # 2.0 (softer)
    student_fg = student[:, 0:num_classes-1]
    student_bg = student[:, -1].unsqueeze(1).repeat(1, num_classes-1)  # no finding is the last prototypes
    student_new = torch.stack((student_bg, student_fg), dim=-1)

    teacher_fg = teacher[:, 0:num_classes-1]
    teacher_bg = teacher[:, -1].unsqueeze(1).repeat(1, num_classes-1)  # no finding is the last prototypes
    teacher_new = torch.stack((teacher_bg, teacher_fg), dim=-1)

    loss_kd = F.kl_div(F.log_softmax(student_new / temperature, dim=2),
                       F.softmax(teacher_new.detach() / temperature, dim=2), reduction='batchmean')
    return loss_kd * temperature * temperature


def distillation_dynamic(student, teacher, temperature=2.0):  # 2.0 (softer)
    student_fg = student[:, 0:num_classes-1]
    student_bg = student[:, -1].unsqueeze(1).repeat(1, num_classes-1)  # no finding is the last prototypes
    student_new = torch.stack((student_bg, student_fg), dim=-1)

    teacher_fg = teacher[0][:, 0:num_classes-1]
    teacher_bg = teacher[0][:, -1].unsqueeze(1).repeat(1, num_classes-1)  # no finding is the last prototypes
    teacher_new = torch.stack((teacher_bg, teacher_fg), dim=-1)
    prob_t1 = F.softmax(teacher_new.detach() / temperature, dim=2)

    teacher_fg = teacher[1][:, 0:num_classes-1]
    teacher_bg = teacher[1][:, -1].unsqueeze(1).repeat(1, num_classes-1)  # no finding is the last prototypes
    teacher_new = torch.stack((teacher_bg, teacher_fg), dim=-1)
    prob_t2 = F.softmax(teacher_new.detach() / temperature, dim=2)

    gamma = np.random.beta(1.5, 1.5)
    prob_t = prob_t1 * gamma + prob_t2 * (1 - gamma)
    loss_kd = F.kl_div(F.log_softmax(student_new / temperature, dim=2), prob_t, reduction='batchmean')

    return loss_kd * temperature * temperature


def proto_mining(model, image, s_simi, t_simi, target, top_k=5):
    th_fg = 0.8    # 0.9
    batch_size = s_simi.shape[0]    # (batch, num_p, h, w)
    num_prototypes_per_class = model.module.num_prototypes // model.module.num_classes   # 50

    assert(top_k <= num_prototypes_per_class)

    if s_simi.shape[-1] > t_simi.shape[-1] or s_simi.shape[-2] > t_simi.shape[-2]:
       t_simi = F.interpolate(t_simi, size=s_simi.shape[2:], mode='bilinear')

    t_simi_max = F.max_pool2d(t_simi, kernel_size=(t_simi.size()[2], t_simi.size()[3])).squeeze(-1).squeeze(-1)  # (batch, num_p)
    s_simi_max = F.max_pool2d(s_simi, kernel_size=(s_simi.size()[2], s_simi.size()[3])).squeeze(-1).squeeze(-1)  # (batch, num_p)

    loss = []
    n_use = []
    for b in range(batch_size):
        real_labels = torch.where(target[b] == 1)[0]
        if False:
        # if len(real_labels) == 1 and real_labels == num_classes-1:   # ignore background class
            loss.append(torch.tensor(0.0).cuda())
        else:
            loss_one_label = []
            for one_label in real_labels:
                prototypes_of_correct_class = torch.t(model.module.prototype_class_identity[:, one_label]).cuda()    # (batch, num_p)
                t_simi_max_one = t_simi_max[b].clone()
                s_simi_max_one = s_simi_max[b].clone()
                t_simi_max_one[prototypes_of_correct_class == 0] = -1e-5
                s_simi_max_one[prototypes_of_correct_class == 0] = -1e-5
                _, t_index = torch.sort(t_simi_max_one, descending=True)
                _, s_index = torch.sort(s_simi_max_one, descending=True)
                loss_temp = torch.tensor(0.0).cuda()
                n_p = 0.
                for p_ind in range(top_k):
                    t_simi_temp = t_simi[b, t_index[p_ind]]
                    t_mask = (t_simi_temp >= th_fg) * (t_simi_temp == t_simi_temp.max())  # (h, w)
                    # t_mask = (t_simi_temp >= th_fg) * (t_simi_temp >= 0.95*t_simi_temp.max())  # (h, w)
                    if t_mask.sum() > 1:
                        ind_pos = torch.where(t_mask == 1)
                        ind = np.random.choice(range(len(ind_pos[0])), 1)
                        t_mask = torch.zeros_like(t_mask)
                        t_mask[ind_pos[0][ind], ind_pos[1][ind]] = 1
                        loss_temp += (s_simi[b, s_index[p_ind]] * t_mask).sum()
                        # loss_temp += (s_simi[b, s_index[p_ind]] * t_mask).sum() / (t_mask.sum() + 1e-5)    # mean
                        n_p += 1
                    elif t_mask.sum() == 1:
                        loss_temp += (s_simi[b, s_index[p_ind]] * t_mask).sum()
                        n_p += 1
                    else:
                        pass
                        # print('batch', b, 'ind', p_ind, "Not meet threshold!")
                loss_one_label.append(loss_temp / (n_p + 1e-5))
                n_use.append(n_p)
            loss.append(sum(loss_one_label))
            # loss.append(sum(loss_one_label)/len(loss_one_label))   # mean

    loss_mining = sum(loss) / batch_size
    n_use = sum(n_use) / (len(n_use) + 1e-5)

    return loss_mining, n_use


def _training(model, dataloader, optimizer=None, train_scale='high', train_last=False, coefs=None, epoch=0, log=print):
    '''
    model: the multi-gpu model
    dataloader:
    optimizer: if None, will be test evaluation
    '''
    start = time.time()
    n_examples = 0
    n_correct = 0
    n_batches = 0
    total_cross_entropy = 0
    total_cluster_cost = 0
    total_separation_cost = 0

    predictions = []
    all_targets = []

    for i, (image, label, _) in enumerate(dataloader):

        input = image.cuda()
        target = label.cuda()

        output_all, min_distances_all, similarities_all = model(input)

        if train_scale == 'high':
            output = output_all[0]
            min_distances = min_distances_all[0]
        elif train_scale == 'middle':
            output = output_all[1]
            min_distances = min_distances_all[1]
        else:
            output = output_all[2]
            min_distances = min_distances_all[2]

        # compute loss
        output_fg = output[:, 0:num_classes-1]
        output_bg = output[:, -1].unsqueeze(1).repeat(1, num_classes-1)  # no finding is the last prototypes
        output_new = torch.stack((output_bg, output_fg), dim=-1)
        cross_entropy = F.cross_entropy(output_new.permute(0, 2, 1), target[:, 0:num_classes-1])

        # calculate cluster and separation cost
        cluster_cost, separation_cost = cluster_sep_loss_fn(model, min_distances, label)

        if train_scale == 'high':
            distil_cost = torch.tensor(0.0).cuda()
            mining_cost = torch.tensor(0.0).cuda()
            n_use = 0.0
        elif train_scale == 'middle':
            distil_cost = distillation_single(student=output, teacher=output_all[0])
            mining_cost, n_use = proto_mining(model, image, s_simi=similarities_all[1], t_simi=similarities_all[0], target=target, top_k=10)
        else:
            distil_cost = distillation_dynamic(student=output, teacher=(output_all[0], output_all[1]))
            mining_cost_h, n_use_h = proto_mining(model, image, s_simi=similarities_all[2], t_simi=similarities_all[0], target=target, top_k=10)
            mining_cost_m, n_use_m = proto_mining(model, image, s_simi=similarities_all[2], t_simi=similarities_all[1], target=target, top_k=10)
            mining_cost = (mining_cost_m + mining_cost_h) / 2.0
            n_use = (n_use_m + n_use_m) / 2.0

        coefs_mining = coefs['distil'] if epoch >= 10 else 0.0    # 6

        if coefs is not None:
            loss = (
                    coefs['crs_ent'] * cross_entropy
                    + coefs['clst'] * cluster_cost
                    + coefs['sep'] * F.relu(2.0 - separation_cost)
                    + coefs_mining * mining_cost
                    + coefs['distil'] * distil_cost
                   )
        else:
            loss = cross_entropy + 0.8 * cluster_cost - 0.08 * separation_cost

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # evaluation statistics
        predicted = torch.argmax(output_new.data, dim=-1)
        n_examples += target.shape[0] * target[:, 0:-1].shape[1]
        n_correct += (predicted == target[:, 0:-1]).sum().item()

        predictions.append(F.softmax(output_new.data, dim=2)[:, :, 1].cpu().numpy())
        all_targets.append(label.numpy())

        n_batches += 1
        total_cross_entropy += cross_entropy.item()
        total_cluster_cost += cluster_cost.item()
        total_separation_cost += separation_cost.item()

        # clip FC weights to ensure positive connections
        #####################################################################
        if train_last:
            prototype_class_identity = model.module.prototype_class_identity.t()
            if train_scale == 'high':
                weight = model.module.last_layer_high.weight.data
                weight[prototype_class_identity == 0] = 0  # set negative weight to be 0
                weight = torch.clamp(weight, min=0.0)  # set positive weight to be more than 0
                model.module.last_layer_high.weight.data = weight
            elif train_scale == 'middle':
                weight = model.module.last_layer_middle.weight.data
                weight[prototype_class_identity == 0] = 0  # set negative weight to be 0
                weight = torch.clamp(weight, min=0.0)  # set positive weight to be more than 0
                model.module.last_layer_middle.weight.data = weight
            elif train_scale == 'low':
                weight = model.module.last_layer_low.weight.data
                weight[prototype_class_identity == 0] = 0  # set negative weight to be 0
                weight = torch.clamp(weight, min=0.0)  # set positive weight to be more than 0
                model.module.last_layer_low.weight.data = weight
            else:
                raise Exception('other level NOT implemented')
        #####################################################################

        if i % 200 == 0:
            print(
                '{} {} \tLoss_total: {:.4f} \tLoss_CE: {:.4f} \tLoss_clust: {:.4f} \tLoss_sepa: {:.4f}'
                '\tLoss_distil: {:.4f} \tLoss_mining: {:.4f} \tAcc: {:.1f} \tn_use: {:.1f}'.format(
                    i, len(dataloader), loss.item(), cross_entropy.item(),
                    cluster_cost.item(), separation_cost.item(), distil_cost.item(), mining_cost.item(),
                    n_correct / (n_examples + 0.000001) * 100, n_use,
                ))
            
            wandb.log({
                "Train Total Loss": loss.item(),
                "Train CE Loss": cross_entropy.item(),
                "Train Cluster Loss": cluster_cost.item(),
                "Train Separation Loss": separation_cost.item(),
                "Train Distillation Loss": distil_cost.item(),
                "Train Mining Loss": mining_cost.item(),
                "Train Mining Coef": coefs_mining,
                "n_use": n_use,
            })

        del input
        del target
        del output
        del predicted
        del min_distances

    end = time.time()

    all_targets = np.concatenate(all_targets, axis=0)
    predictions = np.concatenate(predictions, axis=0)

    all_auc = np.asarray(
        [roc_auc_score(all_targets[:, i], predictions[:, i]) for i in range(num_classes - 1)],
    )
    mean_auc = all_auc.mean()


    log('\t##############TRAIN################')
    log('\ttime: \t{0}'.format(end - start))
    log('\taccu: \t\t{0}%'.format(n_correct / n_examples * 100))
    log('\tAUC: \t\t{0}%'.format(np.around(all_auc, 4) * 100))
    log('\tMean AUC: \t\t{0}%'.format(mean_auc * 100))

    log('\tcross ent: \t{0}'.format(total_cross_entropy / n_batches))
    log('\tcluster: \t{0}'.format(total_cluster_cost / n_batches))
    log('\tseparation:\t{0}'.format(total_separation_cost / n_batches))
    log('\tl1: \t\t{0}'.format(model.module.last_layer_high.weight.norm(p=1).item()))
    log('\t##############TRAIN################')
    
    wandb.log({
        "Train Mean AUC": mean_auc * 100,
    })

    return mean_auc


def _testing(model, dataloader, optimizer=None, coefs=None, log=print):
    '''
    model: the multi-gpu model
    dataloader:
    optimizer: if None, will be test evaluation
    '''
    start = time.time()
    n_examples = 0
    n_correct = 0
    n_batches = 0
    total_cross_entropy_high = 0
    total_cross_entropy_middle = 0
    total_cross_entropy_low = 0

    total_cluster_cost_high = 0
    total_cluster_cost_middle = 0
    total_cluster_cost_low = 0

    total_separation_cost_high = 0
    total_separation_cost_middle = 0
    total_separation_cost_low = 0

    predictions_high = []
    predictions_middle = []
    predictions_low = []
    predictions_comb = []
    all_targets = []

    for i, (image, label, _) in enumerate(dataloader):

        input = image.cuda()
        target = label.cuda()

        with torch.no_grad():

            output, min_distances, _ = model(input)

            # compute loss
            output_fg_high = output[0][:, 0:num_classes-1]
            output_bg_high = output[0][:, -1].unsqueeze(1).repeat(1, num_classes-1)  # no finding is the last prototypes
            output_new_high = torch.stack((output_bg_high, output_fg_high), dim=-1)
            cross_entropy_high = F.cross_entropy(output_new_high.permute(0, 2, 1), target[:, 0:num_classes-1])

            output_fg_middle = output[1][:, 0:num_classes-1]
            output_bg_middle = output[1][:, -1].unsqueeze(1).repeat(1, num_classes-1)  # no finding is the last prototypes
            output_new_middle = torch.stack((output_bg_middle, output_fg_middle), dim=-1)
            cross_entropy_middle = F.cross_entropy(output_new_middle.permute(0, 2, 1), target[:, 0:num_classes-1])

            output_fg_low = output[2][:, 0:num_classes-1]
            output_bg_low = output[2][:, -1].unsqueeze(1).repeat(1, num_classes-1)  # no finding is the last prototypes
            output_new_low = torch.stack((output_bg_low, output_fg_low), dim=-1)
            cross_entropy_low = F.cross_entropy(output_new_low.permute(0, 2, 1), target[:, 0:num_classes-1])

            cluster_cost_high, separation_cost_high = cluster_sep_loss_fn(model, min_distances[0], label)
            cluster_cost_middle, separation_cost_middle = cluster_sep_loss_fn(model, min_distances[1], label)
            cluster_cost_low, separation_cost_low = cluster_sep_loss_fn(model, min_distances[2], label)

            predictions_high.append(F.softmax(output_new_high.data, dim=2)[:, :, 1].cpu().numpy())
            predictions_middle.append(F.softmax(output_new_middle.data, dim=2)[:, :, 1].cpu().numpy())
            predictions_low.append(F.softmax(output_new_low.data, dim=2)[:, :, 1].cpu().numpy())
            predictions_comb.append(F.softmax((output_new_high + output_new_middle + output_new_low), dim=2)[:, :, 1].cpu().numpy())
            all_targets.append(label.numpy())

            n_batches += 1
            total_cross_entropy_high += cross_entropy_high.item()
            total_cross_entropy_middle += cross_entropy_middle.item()
            total_cross_entropy_low += cross_entropy_low.item()
            total_cluster_cost_high += cluster_cost_high.item()
            total_cluster_cost_middle += cluster_cost_middle.item()
            total_cluster_cost_low += cluster_cost_low.item()
            total_separation_cost_high += separation_cost_high.item()
            total_separation_cost_middle += separation_cost_middle.item()
            total_separation_cost_low += separation_cost_low.item()

        del input
        del target
        del output
        del min_distances

    end = time.time()

    predictions_high = np.concatenate(predictions_high, axis=0)      # prob
    predictions_middle = np.concatenate(predictions_middle, axis=0)
    predictions_low = np.concatenate(predictions_low, axis=0)
    predictions_comb = np.concatenate(predictions_comb, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    all_auc_high = np.asarray(
        [roc_auc_score(all_targets[:, i], predictions_high[:, i]) for i in range(num_classes - 1)],
    )
    mean_auc_high = all_auc_high.mean()

    all_auc_middle = np.asarray(
        [roc_auc_score(all_targets[:, i], predictions_middle[:, i]) for i in range(num_classes - 1)],
    )
    mean_auc_middle = all_auc_middle.mean()

    all_auc_low = np.asarray(
        [roc_auc_score(all_targets[:, i], predictions_low[:, i]) for i in range(num_classes - 1)],
    )
    mean_auc_low = all_auc_low.mean()

    all_auc_comb = np.asarray(
        [roc_auc_score(all_targets[:, i], predictions_comb[:, i]) for i in range(num_classes - 1)],
    )
    mean_auc_comb = all_auc_comb.mean()

    log('\t##############TEST################')
    log('\ttime: \t{0}'.format(end - start))
    log('\tcross ent_high: \t{0}'.format(total_cross_entropy_high / n_batches))
    log('\tcross ent_middle: \t{0}'.format(total_cross_entropy_middle / n_batches))
    log('\tcross ent_low: \t{0}'.format(total_cross_entropy_low / n_batches))

    log('\tAUC_high: \t{0}'.format(np.around(all_auc_high, 4) * 100))
    log('\tAUC_middle: \t{0}'.format(np.around(all_auc_middle, 4) * 100))
    log('\tAUC_low:\t{0}'.format(np.around(all_auc_low, 4) * 100))
    log('\tAUC_comb:\t{0}'.format(np.around(all_auc_comb, 4) * 100))

    log('\tMean AUC high: \t{0}'.format(mean_auc_high * 100))
    log('\tMean AUC middle: \t{0}'.format(mean_auc_middle * 100))
    log('\tMean AUC low:\t{0}'.format(mean_auc_low * 100))
    log('\tMean AUC comb:\t{0}'.format(mean_auc_comb * 100))
    log('\t##############TEST################')
    
    wandb.log({
        "Test AUC high": mean_auc_high * 100,
        "Test AUC middle": mean_auc_middle * 100,
        "Test AUC low": mean_auc_low * 100,
        "Test AUC comb": mean_auc_comb * 100,
    })

    return mean_auc_comb


def train(model, dataloader, optimizer, train_scale='high', train_last=False, coefs=None, epoch=0, log=print):
    assert (optimizer is not None)
    assert (train_scale in ['high', 'middle', 'low'])
    log('\ttrain')
    if train_scale == 'high':
        model.train()
    else:
        model.train()
        model.module.features.eval()  # fix BN and Dropout in Backbone
    return _training(model=model, dataloader=dataloader, optimizer=optimizer, train_scale=train_scale,
                     train_last=train_last, coefs=coefs, epoch=epoch, log=log)


def test(model, dataloader, log=print):
    log('\ttest')
    model.eval()
    return _testing(model=model, dataloader=dataloader, optimizer=None, log=log)


def warm_only_high(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = False

    for p in model.module.add_on_layers_high.parameters():
        p.requires_grad = True
    for p in model.module.add_on_layers_middle.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers_low.parameters():
        p.requires_grad = False

    model.module.prototype_vectors_high.requires_grad = True
    model.module.prototype_vectors_middle.requires_grad = False
    model.module.prototype_vectors_low.requires_grad = False

    for p in model.module.last_layer_high.parameters():
        p.requires_grad = False
    for p in model.module.last_layer_middle.parameters():
        p.requires_grad = False
    for p in model.module.last_layer_low.parameters():
        p.requires_grad = False
    log('\twarm high')


def joint_high(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = True

    for p in model.module.add_on_layers_high.parameters():
        p.requires_grad = True
    for p in model.module.add_on_layers_middle.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers_low.parameters():
        p.requires_grad = False

    model.module.prototype_vectors_high.requires_grad = True
    model.module.prototype_vectors_middle.requires_grad = False
    model.module.prototype_vectors_low.requires_grad = False

    for p in model.module.last_layer_high.parameters():
        p.requires_grad = False
    for p in model.module.last_layer_middle.parameters():
        p.requires_grad = False
    for p in model.module.last_layer_low.parameters():
        p.requires_grad = False

    log('\tjoint high')


def last_only_high(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = False

    for p in model.module.add_on_layers_high.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers_middle.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers_low.parameters():
        p.requires_grad = False

    model.module.prototype_vectors_high.requires_grad = False
    model.module.prototype_vectors_middle.requires_grad = False
    model.module.prototype_vectors_low.requires_grad = False

    for p in model.module.last_layer_high.parameters():
        p.requires_grad = True
    for p in model.module.last_layer_middle.parameters():
        p.requires_grad = False
    for p in model.module.last_layer_low.parameters():
        p.requires_grad = False

    log('\tlast layer high')


def joint_middle(model, log=print):
    for name, p in model.module.features.named_parameters():
        if 'latlayer1' in name or 'smooth1' in name:
            p.requires_grad = True
            print('update backbone layer:', name)
        else:
            p.requires_grad = False

    for p in model.module.add_on_layers_high.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers_middle.parameters():
        p.requires_grad = True
    for p in model.module.add_on_layers_low.parameters():
        p.requires_grad = False

    model.module.prototype_vectors_high.requires_grad = False
    model.module.prototype_vectors_middle.requires_grad = True
    model.module.prototype_vectors_low.requires_grad = False

    for p in model.module.last_layer_high.parameters():
        p.requires_grad = False
    for p in model.module.last_layer_middle.parameters():
        p.requires_grad = False
    for p in model.module.last_layer_low.parameters():
        p.requires_grad = False

    log('\tjoint middle')


def last_only_middle(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = False

    for p in model.module.add_on_layers_high.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers_middle.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers_low.parameters():
        p.requires_grad = False

    model.module.prototype_vectors_high.requires_grad = False
    model.module.prototype_vectors_middle.requires_grad = False
    model.module.prototype_vectors_low.requires_grad = False

    for p in model.module.last_layer_high.parameters():
        p.requires_grad = False
    for p in model.module.last_layer_middle.parameters():
        p.requires_grad = True
    for p in model.module.last_layer_low.parameters():
        p.requires_grad = False

    log('\tlast layer middle')


def joint_low(model, log=print):
    for name, p in model.module.features.named_parameters():
        if 'latlayer2' in name or 'smooth2' in name:
            p.requires_grad = True
            print('update backbone layer:', name)
        else:
            p.requires_grad = False

    for p in model.module.add_on_layers_high.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers_middle.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers_low.parameters():
        p.requires_grad = True

    model.module.prototype_vectors_high.requires_grad = False
    model.module.prototype_vectors_middle.requires_grad = False
    model.module.prototype_vectors_low.requires_grad = True

    for p in model.module.last_layer_high.parameters():
        p.requires_grad = False
    for p in model.module.last_layer_middle.parameters():
        p.requires_grad = False
    for p in model.module.last_layer_low.parameters():
        p.requires_grad = False

    log('\tjoint low')


def last_only_low(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = False

    for p in model.module.add_on_layers_high.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers_middle.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers_low.parameters():
        p.requires_grad = False

    model.module.prototype_vectors_high.requires_grad = False
    model.module.prototype_vectors_middle.requires_grad = False
    model.module.prototype_vectors_low.requires_grad = False

    for p in model.module.last_layer_high.parameters():
        p.requires_grad = False
    for p in model.module.last_layer_middle.parameters():
        p.requires_grad = False
    for p in model.module.last_layer_low.parameters():
        p.requires_grad = True

    log('\tlast layer low')
