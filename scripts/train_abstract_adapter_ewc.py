#!/usr/bin/env python

import sys, os
from scripts import train_adapter_ewc as train
from importlib import reload

# -- Set configurations manually -- #
device = 0
nr_samples = 100
save_steps = 15
nr_epochs = 250
lr = 1e-4
steps_per_epoch = 75

mappings = {
            '98': 'Task98_Dryad', '99': 'Task99_HarP', '97': 'Task97_DecathHip',
            '79': 'Task79_UCL', '78': 'Task78_I2CVB', '77': 'Task77_ISBI', '76': 'Task76_DecathProst'
           }
train_on = [
            # ['99'], #['98'], ['97'],
            # ['99', '98'],
            # ['99', '98', '97'],
            ['79'], #['78'], ['77'], ['76'],
            ['79', '78'],
            ['79', '78', '77'],
            ['79', '78', '77', '76']
           ]

continue_ = False
finished = False
continue_with_epoch = 0

# model_type = 'vit_b'
# checkpoint = '/local/scratch/aranem/MIDL_2024/MIDL_2024_raw_data/checkpoints/sam_vit_b_01ec64.pth'

model_type = 'vit_h'
checkpoint = '/media/aranem_locale/AR_subs_exps/MIDL_2024/MIDL_2024_raw_data/checkpoints/sam_vit_h_4b8939.pth'

# -- Train based on the configurations -- #
for tasks in train_on:
    trained_list = []
    for task in tasks:
        prev_mod_built = '_'.join(trained_list)
        trained_list.append(task)
        built_ts = '_'.join(trained_list)
        inp = f'/home/aranem_locale/Desktop/mnts/local/scratch/maflal/{mappings[task]}/{model_type}/Tr'
        val = f'/home/aranem_locale/Desktop/mnts/local/scratch/maflal/{mappings[task]}/{model_type}/Ts'
        out_folder = f'/media/aranem_locale/AR_subs_exps/MIDL_2024/MIDL_2024_trained_models_prost_huge/EWC_frozen_after_first_task/sam_adapter_{built_ts}_{nr_samples}_lr-1e-4-with-adapter-lr-1e-3_ewc'
        # out_folder = f'/media/aranem_locale/AR_subs_exps/MIDL_2024/MIDL_2024_trained_models_hip/EWC/sam_adapter_{built_ts}_{nr_samples}_lr-1e-4-with-adapter-lr-1e-3_ewc'
        # out_folder = f'/home/aranem_locale/Desktop/SAM_CL/trained_models/sam_resnet50_torch_{nr_epochs}_{built_ts}_ce_mse_mse_{nr_samples}_samples_1_2_1_2_wd'

        exp_name = built_ts + "_lr-1e-4-with-adapter-lr-1e-3_ewc_huge"

        # -- Check if it is already trained or not -- #
        if os.path.exists(out_folder):
            # -- Started training on, so restore if more than one checkpoint -- #
            chks = [x for x in os.listdir(out_folder) if '.pt' in x and 'best' not in x and 'adapter' not in x]
            if len(chks) <= 1:  # Only 0000.pt in the list
                if len(trained_list) > 1: # <-- We still need load_model here
                    prev_model = out_folder.replace(built_ts, prev_mod_built)
                    continue_, finished, continue_with_epoch = True, True, 0
                    load_model = os.path.join(prev_model, 'best_model.pt')    # <-- Should exist!
                else:
                    continue_, finished, continue_with_epoch = False, False, 0
            else:
                chks.sort()
                chkp = chks[-1]
                if str(nr_epochs) in chkp:
                    continue_, finished, continue_with_epoch = False, False, 0
                    continue    # <-- Finished with training for this task
                continue_, finished, continue_with_epoch = True, False, int(chkp.split('.pt')[0].replace('_model', ''))
                load_model = os.path.join(out_folder, '%04d_model.pt' % continue_with_epoch)

        elif len(trained_list) > 1: # <-- We still need load_model here
            prev_model = out_folder.replace(built_ts, prev_mod_built)
            continue_, finished, continue_with_epoch = True, True, 0
            load_model = os.path.join(prev_model, 'best_model.pt')    # <-- Should exist!

        # -- Build up arguments -- #
        args = [sys.argv[0], '--train-dir']
        args += [inp]
        args += ['--model-dir', out_folder]
        # if len(trained_list) > 1: # After first task freeze SAM
        #     args += ['--freeze_sam']
        args += ['--task', task]
        if continue_:
            args += ['--load-model', load_model]
            if not finished:
                args += ['--initial-epoch', str(continue_with_epoch)]
        args += ['--gpu', str(device)]
        args += ['--epochs', str(nr_epochs)]
        args += ['--save_steps', str(save_steps)]
        args += ['--model_type', model_type]
        args += ['--checkpoint', checkpoint]
        args += ['--val-dir', val]
        args += ['--lr', str(lr)]
        args += ['--steps-per-epoch', str(steps_per_epoch)]
        args += ['--exp_name', exp_name]

        # -- Train -- #
        sys.argv = args

        train = reload(train)   # So the log files can be updated as well
        train.train()