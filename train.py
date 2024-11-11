import os
import sys
from tqdm import tqdm
from easydict import EasyDict as edict
from argparse import ArgumentParser,Namespace
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from datareader.dataset import collate_fn
from arguments import Config
from utils.system import safe_state,set_seed
from utils.general import train_bn,eval_bn,summarize_loss
import utils.general
from datareader import PackedData
from datareader.dataset import DatasetFactory
from model import FEASAI
from model.loss import TotalLoss
from datetime import datetime


def normalize_to_255(tensor):
    # 确保 tensor 是浮点数类型，以避免整数除法
    tensor = tensor.float()
    min_val = tensor.min()
    max_val = tensor.max()

    # 防止除以 0，进行归一化并缩放到 [0, 255]
    normalized_tensor = 255 * (tensor - min_val) / (max_val - min_val + 1e-8)

    # 返回为 uint8 类型
    return normalized_tensor.to(torch.uint8)  # 转换为 uint8 类型


def prepare_data(opt:Config,world_size:int,rank:int):
    train_dataset = DatasetFactory().get(opt.dataset,opt,"Train")
    train_sampler = DistributedSampler(train_dataset,num_replicas=world_size,rank=rank,shuffle=True,drop_last=False)
    train_dataloader = DataLoader(train_dataset,batch_size=opt.bs,sampler=train_sampler,drop_last=False)
    test_dataset = DatasetFactory().get(opt.dataset,opt,"Test")
    test_sampler = DistributedSampler(test_dataset,num_replicas=world_size,rank=rank,shuffle=True,drop_last=False)
    test_dataloader = DataLoader(test_dataset,batch_size=opt.bs,sampler=test_sampler,drop_last=False)
    return train_dataloader,test_dataloader,train_sampler,test_sampler

def prepare_output(opt:Config, rank:int):
    date = datetime.now().strftime('%Y%m%d-%H%M%S')
    results_dir = os.path.abspath(f'/workspace/xak1wx/FEASAI/results/{opt.exp_name}_{date}')
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "cfg_args.txt"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(opt))))
    tb = SummaryWriter(log_dir=f"{results_dir}/{rank}", flush_secs=60)
    return results_dir, tb


def train_fn(rank,opt:Config):
    # prepare
    # dist.init_process_group('nccl',f'tcp://{opt.ip}:{opt.port}',rank=rank,world_size=opt.world_size)
    # rank = dist.get_rank()

    print(f'rank:{rank} is initialized.')
    train_dataloader,test_dataloader,_,_ = prepare_data(opt,opt.world_size,rank)
    results_dir,tb = prepare_output(opt,rank)
    log_file_path = f"{results_dir}/log.txt"
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    # torch.cuda.set_device(rank)
    # network
    net = FEASAI()
    optimizer = torch.optim.Adam(net.parameters(),lr=opt.lr)
    gamma = (opt.lr_end/opt.lr)**(1./opt.max_epoch)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
    # 分布式可以加上net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net).cuda()
    # net = DDP(net.cuda(),device_ids=[rank])
    net = net.cuda()
    net = net.train()
    # checkpoint
    epoch_start = 0
    # load checkpoint
    ckpt_path = f"{results_dir}/model/checkpoint.pth"
    if os.path.exists(ckpt_path):
        cuda_map = {f"cuda:0":f"cuda:{rank}"}
        checkpoint = torch.load(ckpt_path,map_location=cuda_map)
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        epoch_start = checkpoint['epoch']
    criterion = TotalLoss()
    ## train
    with open(log_file_path, "a") as log_file:
        log_file.write(f"Start training with config: {opt}\n")
    # 训练和验证过程
    for epoch in range(opt.max_epoch + 1):
        if epoch <= epoch_start:
            continue
        with open(log_file_path, "a") as log_file:
            net.apply(train_bn) if epoch <= opt.fix_bn_epoch else net.apply(eval_bn)

            # Train
            with tqdm(total=len(train_dataloader)) as pbar:
                pbar.set_description_str(f'[epoch {epoch}|Train]')
                net = net.train()
                # train_sampler.set_epoch(epoch)
                loss_recorder, metric = edict(all=0.), edict(psnr=0., ssim=0.)

                for i, var in enumerate(train_dataloader):
                    net.zero_grad()
                    optimizer.zero_grad()
                    var: PackedData = PackedData().set(var).cuda()
                    var.epoch = epoch
                    var = net(var)
                    loss_keys, losses = criterion(var, epoch=epoch)
                    loss = summarize_loss(opt.loss_weights, losses)
                    loss.backward()
                    optimizer.step()

                    psnr, ssim = utils.general.psnr(var.pred_frame, var.gt), utils.general.ssim(var.pred_frame, var.gt)
                    pbar.set_postfix_str(f"loss:{loss.item():.4f},psnr:{psnr:.4f},ssim:{ssim:.4f}")
                    pbar.update(1)

                    # 记录
                    metric.psnr += psnr
                    metric.ssim += ssim
                    loss_recorder.all += loss.item()
                    for key, loss in zip(loss_keys, losses):
                        if key not in loss_recorder.keys():
                            loss_recorder[key] = 0.
                        loss_recorder[key] += loss.item()

                # 写入训练日志
                tb.add_scalar(f"train/lr", optimizer.param_groups[0]['lr'], epoch)
                n_samples = len(train_dataloader)
                for key in loss_recorder.keys():
                    tb.add_scalar(f"train/loss_{key}", loss_recorder[key] / n_samples, epoch)
                    print(f"train/loss_{key}", loss_recorder[key] / n_samples)
                print(f"train/psnr", metric.psnr / n_samples)

                tb.add_scalar(f"train/psnr", metric.psnr / n_samples, epoch)
                tb.add_scalar(f"train/ssim", metric.ssim / n_samples, epoch)
                log_file.write(f"[epoch {epoch}|train]: average loss: {loss_recorder.all / n_samples:.4f}, "
                               f"average psnr: {metric.psnr / n_samples:.4f}, "
                               f"average ssim: {metric.ssim / n_samples:.4f}\n")
                print(f"[epoch {epoch}|train]: average loss: {loss_recorder.all / n_samples:.4f}.")
                print(f"[epoch {epoch}|train]: average psnr: {metric.psnr / n_samples:.4f}.")
                print(f"[epoch {epoch}|train]: average ssim: {metric.ssim / n_samples:.4f}.")

                # View
                with torch.no_grad():
                    for i in range(1):
                        pred_frame_norm = normalize_to_255(var.pred_frame.cpu()[i, ...])
                        tb.add_image(f"train/pred_{i:04d}", pred_frame_norm, epoch)
                        gt_norm = normalize_to_255(var.gt.cpu()[i, ...])
                        tb.add_image(f"train/gt_{i:04d}", gt_norm, epoch)
                        acc_frame_norm = normalize_to_255(var.ev_ref_frame.cpu()[i, ...])
                        tb.add_image(f"train/acc_event_{i:04d}", acc_frame_norm, epoch)
                        occ_frame_norm = normalize_to_255(var.img_ref_frame.cpu()[i, ...])
                        tb.add_image(f"train/acc_frame_{i:04d}", occ_frame_norm, epoch)
                        pred_depth_ev_norm = normalize_to_255(var.ev_depth_frame.cpu()[i, ...])
                        tb.add_image(f"train/depth_ev_{i:04d}", pred_depth_ev_norm, epoch)
                        pred_depth_img_norm = normalize_to_255(var.img_depth_frame.cpu()[i, ...])
                        tb.add_image(f"train/depth_img_{i:04d}", pred_depth_img_norm, epoch)
                        gt_depth_norm = normalize_to_255(var.gt_depth_frame.cpu()[i, ...])
                        tb.add_image(f"train/depth_gt_{i:04d}", gt_depth_norm, epoch)
                scheduler.step()

            # Eval
            with torch.no_grad():
                with tqdm(total=len(test_dataloader)) as pbar:
                    pbar.set_description_str(f'[epoch {epoch}|Val]')
                    net = net.eval()
                    # test_sampler.set_epoch(epoch)
                    loss_recorder, metric = edict(all=0.), edict(psnr=0., ssim=0.)

                    for i, var in enumerate(test_dataloader):
                        var: PackedData = PackedData().set(var).cuda()
                        var = net(var)
                        loss_keys, losses = criterion(var, epoch=epoch)
                        loss = summarize_loss(opt.loss_weights, losses)
                        psnr, ssim = utils.general.psnr(var.pred_frame, var.gt), utils.general.ssim(var.pred_frame, var.gt)
                        pbar.set_postfix_str(f"loss:{loss.item():.4f},psnr:{psnr:.4f},ssim:{ssim:.4f}")
                        pbar.update(1)

                        # 记录
                        metric.psnr += psnr
                        metric.ssim += ssim
                        loss_recorder.all += loss.item()
                        for key, loss in zip(loss_keys, losses):
                            if key not in loss_recorder.keys():
                                loss_recorder[key] = 0.
                            loss_recorder[key] += loss.item()

                        if rank == 0 and i < 2:
                            # 对 val/pred_frame 进行归一化
                            pred_frame_norm = normalize_to_255(var.pred_frame.cpu()[0, ...])
                            tb.add_image(f"val/pred_{i:04d}", pred_frame_norm, epoch)

                            # 对 val/gt 进行归一化
                            gt_norm = normalize_to_255(var.gt.cpu()[0, ...])
                            tb.add_image(f"val/gt_{i:04d}", gt_norm, epoch)

                            # 对 train/acc_event 进行归一化
                            acc_event_norm = normalize_to_255(var.ev_ref_frame.cpu()[0, ...])
                            tb.add_image(f"val/acc_event_{i:04d}", acc_event_norm, epoch)

                            # 对 train/occ_frame 进行归一化
                            occ_frame_norm = normalize_to_255(var.img_ref_frame.cpu()[0, ...])
                            tb.add_image(f"val/acc_frame_{i:04d}", occ_frame_norm, epoch)

                            # 对 val/gt_depth 进行归一化
                            gt_depth_norm = normalize_to_255(var.gt_depth_frame.cpu()[0, ...])
                            tb.add_image(f"val/depth_gt_{i:04d}", gt_depth_norm, epoch)

                            pred_depth_ev_norm = normalize_to_255(var.ev_depth_frame.cpu()[0, ...])
                            tb.add_image(f"val/depth_ev_{i:04d}", pred_depth_ev_norm, epoch)
                            pred_depth_img_norm = normalize_to_255(var.img_depth_frame.cpu()[0, ...])
                            tb.add_image(f"val/depth_img_{i:04d}", pred_depth_img_norm, epoch)


                    n_val_samples = len(test_dataloader)
                    # 写入验证日志
                    log_file.write(f"[epoch {epoch}|val]: average loss: {loss_recorder.all / n_val_samples:.4f}, "
                                   f"average psnr: {metric.psnr / n_val_samples:.4f}, "
                                   f"average ssim: {metric.ssim / n_val_samples:.4f}\n")
                    for key in loss_recorder.keys():
                        tb.add_scalar(f"val/loss_{key}", loss_recorder[key] / n_val_samples, epoch)
                    tb.add_scalar(f"val/psnr", metric.psnr / n_val_samples, epoch)
                    tb.add_scalar(f"val/ssim", metric.ssim / n_val_samples, epoch)
                    print(f"[epoch {epoch}|val]: average loss: {loss_recorder.all / n_val_samples:.4f}.")
                    print(f"[epoch {epoch}|val]: average psnr: {metric.psnr / n_val_samples:.4f}.")
                    print(f"[epoch {epoch}|val]: average ssim: {metric.ssim / n_val_samples:.4f}.")

        # save model
        if rank == 0 and epoch % opt.per_save_model_epoch ==0:
            os.makedirs(f"{results_dir}/model",exist_ok=True)
            checkpoint = dict(
                net = net.state_dict(),
                optimizer = optimizer.state_dict(),
                scheduler = scheduler.state_dict(),
                epoch = epoch,
            )
            torch.save(checkpoint,f"{results_dir}/model/epoch_{epoch:04d}.pth")
        # sync
        # dist.barrier()
    tb.close()

if __name__ == '__main__':
    #
    parser = ArgumentParser(description="Training script parameters")
    config = Config(parser)
    parser.add_argument("--batch_size", "-b", type=int, default=-1, help="batch_size")
    args = parser.parse_args(sys.argv[1:])
    config.extract(args)
    safe_state(config.quiet)
    set_seed(config.seed)
    torch.autograd.set_detect_anomaly(config.detect_anomaly)
    # mp.spawn(train_fn,nprocs=config.world_size,args=(config,))
    if args.batch_size != -1:
        config.bs = args.batch_size
    train_fn(0,config)