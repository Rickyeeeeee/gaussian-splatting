import os

scenes = ['Courthouse', 'Truck', 'Caterpillar', 'Barn', 'Meetingroom', 'Ignatius']
data_devices = ['cpu', 'cuda', 'cuda','cuda','cuda', 'cuda']
data_base_path='/workspace/data/Datasets/tnt_dataset/tnt'
out_base_path='/workspace/work/Outputs/tnt'
out_name='3dgs'
gpu_id=0

for id, scene in enumerate(scenes):

#     cmd = f'rm -rf {out_base_path}/{out_name}/{scene}/*'
#     print(cmd)
#     os.system(cmd)
    
    # create folder name 0
    # cmd = f'mkdir -p {data_base_path}/{scene}/sparse/0'
    # print(cmd)
    # os.system(cmd)
    # cmd = f'cp {data_base_path}/{scene}/sparse/cameras.bin {data_base_path}/{scene}/sparse/0/cameras.bin'
    # print(cmd)
    # os.system(cmd)
    # cmd = f'cp {data_base_path}/{scene}/sparse/images.bin {data_base_path}/{scene}/sparse/0/images.bin'
    # print(cmd)
    # os.system(cmd)
    # cmd = f'cp {data_base_path}/{scene}/sparse/points3D.bin {data_base_path}/{scene}/sparse/0/points3D.bin'
    # print(cmd)
    # os.system(cmd)
    
#     common_args = f" --disable_viewer --quiet -r2 --data_device {data_devices[id]} --eval"
#     cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python train.py -s {data_base_path}/{scene} -m {out_base_path}/{out_name}/{scene} {common_args}'
#     print(cmd)
#     os.system(cmd)

    common_args = f" --quiet --eval --skip_train --iteration 30000"
    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python render.py -s {data_base_path}/{scene} -m {out_base_path}/{out_name}/{scene} --data_device {data_devices[id]} {common_args}'
    print(cmd)
    os.system(cmd)

    cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} python metrics.py " + \
          f"-m {out_base_path}/{out_name}/{scene} "
    print(cmd)
    os.system(cmd)

    # break