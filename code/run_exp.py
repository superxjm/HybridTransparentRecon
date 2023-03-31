import os

config_files = ['cat_2_config.conf', 'cat_2_config_wo_corr.conf', 'cat_2_config_wo_ncorr.conf', 'cat_2_config_wo_color.conf',
                'cat_2_config_wo_sil.conf', 'cat_2_config_wo_reg.conf']

epoch_num = 300
for config_file in config_files:
    script_str = 'python training/exp_runner.py --conf confs/%s --train_displacements_multi_mlp --use_our_corr --vsa_num 150 --batch_size=1 --expname %s --nepoch %d ' % (config_file, config_file, epoch_num) 
    print(script_str)
    os.system(script_str)
