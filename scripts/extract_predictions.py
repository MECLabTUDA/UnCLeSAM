import sys, os
import numpy as np
from tqdm import tqdm
from scripts import segment
from importlib import reload

ins = [
    
      #  '/home/aranem_locale/Desktop/mnts/local/scratch/maflal/Task99_HarP/vit_b/Ts',
      #  '/home/aranem_locale/Desktop/mnts/local/scratch/maflal/Task98_Dryad/vit_b/Ts',
      #  '/home/aranem_locale/Desktop/mnts/local/scratch/maflal/Task97_DecathHip/vit_b/Ts',

       '/home/aranem_locale/Desktop/mnts/local/scratch/maflal/Task79_UCL/vit_b/Ts',
       '/home/aranem_locale/Desktop/mnts/local/scratch/maflal/Task78_I2CVB/vit_b/Ts',
       '/home/aranem_locale/Desktop/mnts/local/scratch/maflal/Task77_ISBI/vit_b/Ts',
       '/home/aranem_locale/Desktop/mnts/local/scratch/maflal/Task76_DecathProst/vit_b/Ts',
     ]

models = [
          #  '/media/aranem_locale/AR_subs_exps/MIDL_2024/MIDL_2024_trained_models_hip/EWC_frozen_after_first_task/sam_adapter_99_98_97_100_lr-1e-4-with-adapter-lr-1e-3_ewc/best_model.pt',
          #  '/media/aranem_locale/AR_subs_exps/MIDL_2024/MIDL_2024_trained_models_hip/EWC_frozen_after_first_task/sam_adapter_99_98_100_lr-1e-4-with-adapter-lr-1e-3_ewc/best_model.pt',
          #  '/media/aranem_locale/AR_subs_exps/MIDL_2024/MIDL_2024_trained_models_hip/EWC_frozen_after_first_task/sam_adapter_99_100_lr-1e-4-with-adapter-lr-1e-3_ewc/best_model.pt',
           
          #  '/media/aranem_locale/AR_subs_exps/MIDL_2024/MIDL_2024_trained_models_prost/EWC_frozen_after_first_task/sam_adapter_79_78_77_76_100_lr-1e-4-with-adapter-lr-1e-3_ewc/best_model.pt',
          #  '/media/aranem_locale/AR_subs_exps/MIDL_2024/MIDL_2024_trained_models_prost/EWC_frozen_after_first_task/sam_adapter_79_78_77_100_lr-1e-4-with-adapter-lr-1e-3_ewc/best_model.pt',
          #  '/media/aranem_locale/AR_subs_exps/MIDL_2024/MIDL_2024_trained_models_prost/EWC_frozen_after_first_task/sam_adapter_79_78_100_lr-1e-4-with-adapter-lr-1e-3_ewc/best_model.pt',
          #  '/media/aranem_locale/AR_subs_exps/MIDL_2024/MIDL_2024_trained_models_prost/EWC_frozen_after_first_task/sam_adapter_79_100_lr-1e-4-with-adapter-lr-1e-3_ewc/best_model.pt',
        
           '/media/aranem_locale/AR_subs_exps/MIDL_2024/MIDL_2024_trained_models_prost_huge/Seq/sam_adapter_79_100_lr-1e-4-with-adapter-lr-1e-3/best_model.pt',
           '/media/aranem_locale/AR_subs_exps/MIDL_2024/MIDL_2024_trained_models_prost_huge/Seq/sam_adapter_78_100_lr-1e-4-with-adapter-lr-1e-3/best_model.pt',
           '/media/aranem_locale/AR_subs_exps/MIDL_2024/MIDL_2024_trained_models_prost_huge/Seq/sam_adapter_77_100_lr-1e-4-with-adapter-lr-1e-3/best_model.pt',
           '/media/aranem_locale/AR_subs_exps/MIDL_2024/MIDL_2024_trained_models_prost_huge/Seq/sam_adapter_76_100_lr-1e-4-with-adapter-lr-1e-3/best_model.pt',
           '/media/aranem_locale/AR_subs_exps/MIDL_2024/MIDL_2024_trained_models_prost_huge/Seq/sam_adapter_79_78_100_lr-1e-4-with-adapter-lr-1e-3/best_model.pt',
           '/media/aranem_locale/AR_subs_exps/MIDL_2024/MIDL_2024_trained_models_prost_huge/Seq/sam_adapter_79_78_77_100_lr-1e-4-with-adapter-lr-1e-3/best_model.pt',
           '/media/aranem_locale/AR_subs_exps/MIDL_2024/MIDL_2024_trained_models_prost_huge/Seq/sam_adapter_79_78_77_76_100_lr-1e-4-with-adapter-lr-1e-3/best_model.pt',

        #   '/home/aranem_locale/Desktop/mnts/local/scratch/aranem/MIDL_2024/MIDL_2024_trained_models_SAM_always_frozen/Seq/sam_adapter_99_100_lr-1e-4-with-adapter-lr-1e-3/best_model.pt',
          # '/home/aranem_locale/Desktop/mnts/local/scratch/aranem/MIDL_2024/MIDL_2024_trained_models_SAM_always_frozen/Seq/sam_adapter_98_100_lr-1e-4-with-adapter-lr-1e-3/best_model.pt',
          # '/home/aranem_locale/Desktop/mnts/local/scratch/aranem/MIDL_2024/MIDL_2024_trained_models_SAM_always_frozen/Seq/sam_adapter_97_100_lr-1e-4-with-adapter-lr-1e-3/best_model.pt',
        #   '/home/aranem_locale/Desktop/mnts/local/scratch/aranem/MIDL_2024/MIDL_2024_trained_models_SAM_always_frozen/Seq/sam_adapter_98_100_lr-1e-4-with-adapter-lr-1e-3/best_model.pt',
        #   '/home/aranem_locale/Desktop/mnts/local/scratch/aranem/MIDL_2024/MIDL_2024_trained_models_SAM_always_frozen/Seq/sam_adapter_97_100_lr-1e-4-with-adapter-lr-1e-3/best_model.pt',
        #   '/home/aranem_locale/Desktop/mnts/local/scratch/aranem/MIDL_2024/MIDL_2024_trained_models_SAM_always_frozen/Seq/sam_adapter_99_98_100_lr-1e-4-with-adapter-lr-1e-3/best_model.pt',
        #   '/home/aranem_locale/Desktop/mnts/local/scratch/aranem/MIDL_2024/MIDL_2024_trained_models_SAM_always_frozen/Seq/sam_adapter_99_98_97_100_lr-1e-4-with-adapter-lr-1e-3/best_model.pt',
          # '/home/aranem_locale/Desktop/mnts/local/scratch/aranem/MIDL_2024/MIDL_2024_trained_models_SAM_always_frozen/EWC/sam_adapter_99_100_lr-1e-4-with-adapter-lr-1e-3_ewc/best_model.pt',
          # '/home/aranem_locale/Desktop/mnts/local/scratch/aranem/MIDL_2024/MIDL_2024_trained_models_SAM_always_frozen/EWC/sam_adapter_99_98_100_lr-1e-4-with-adapter-lr-1e-3_ewc/best_model.pt',
          # '/home/aranem_locale/Desktop/mnts/local/scratch/aranem/MIDL_2024/MIDL_2024_trained_models_SAM_always_frozen/EWC/sam_adapter_99_98_97_100_lr-1e-4-with-adapter-lr-1e-3_ewc/best_model.pt',
          # '/home/aranem_locale/Desktop/mnts/local/scratch/aranem/MIDL_2024/MIDL_2024_trained_models_SAM_always_frozen/RWalk/sam_adapter_99_100_lr-1e-4-with-adapter-lr-1e-3_rwalk/best_model.pt',
          # '/home/aranem_locale/Desktop/mnts/local/scratch/aranem/MIDL_2024/MIDL_2024_trained_models_SAM_always_frozen/RWalk/sam_adapter_99_98_100_lr-1e-4-with-adapter-lr-1e-3_rwalk/best_model.pt',
          # '/home/aranem_locale/Desktop/mnts/local/scratch/aranem/MIDL_2024/MIDL_2024_trained_models_SAM_always_frozen/RWalk/sam_adapter_99_98_97_100_lr-1e-4-with-adapter-lr-1e-3_rwalk/best_model.pt',
          ]

# Extract predictions
for model in models:
    # epoch = int(model.split(os.sep)[-1][:-9].replace('_model', ''))
    for inp in ins:
        res_HD, res_Dice, res_IoU = [], [], []
        print(f"Creating predictions with {model.split(os.sep)[-2]} for {inp.split(os.sep)[-3]}:")
        out_ = os.path.join(os.path.sep, *model.split(os.path.sep)[:-1], 'inference', inp.split(os.sep)[-3])
        cases = [x[:-4] for x in os.listdir(inp) if '._' not in x and '.json' not in x and 'DS_Store' not in x and '.npz' in x]
        cases = np.unique(cases)
        # for case in cases:
        for case in tqdm(cases):
            npz = os.path.join(inp, case+'.npz')
            out = os.path.join(out_, case)
            os.makedirs(out, exist_ok=True)

            # -- Build up arguments and do registration -- #
            args = [sys.argv[0], '--model']
            args += [model, '--npz']
            args += [npz]
            args += ['--out', out]
            args += ['--log', os.path.join(out_, "inference_log.txt")]
            sys.argv = args

            segment = reload(segment)   # So the log files can be updated as well
            HD, Dice, IoU = segment.segment()
            # HD, Dice, IoU = segment_single()
            if HD != np.inf:
                res_HD.append(HD)
                res_Dice.append(Dice)
                res_IoU.append(IoU)
        print(f"Performance of model {model.split(os.sep)[-2]} for {inp.split(os.sep)[-2]} (HD -- Dice -- IoU): {np.mean(res_HD):.2f} -- {np.mean(res_Dice):.2f}% -- {np.mean(res_IoU):.2f}%")
