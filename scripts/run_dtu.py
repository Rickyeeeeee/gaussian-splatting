import os

scenes = [24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110, 114, 118, 122]
data_base_path='/workspace/data/Datasets/dtu/DTU'
out_base_path='/workspace/work/Outputs/dtu'
eval_path='/workspace/data/replica_sclike_colmap_dnsplatter/dtu_dataset/MVS_Data'
out_name='3dgs'
gpu_id=0

for scene in scenes:
#     cmd = f'rm -rf {out_base_path}/{out_name}/dtu_scan{scene}/*'
#     print(cmd)
#     os.system(cmd)

#     cmd = f'cp -rf {data_base_path}/scan{scene}/sparse/0/* {data_base_path}/scan{scene}/sparse/'
#     print(cmd)
#     os.system(cmd)

#     common_args = "  --disable_viewer --quiet -r2 --eval"
#     cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python train.py -s {data_base_path}/scan{scene} -m {out_base_path}/{out_name}/dtu_scan{scene} {common_args}'
#     print(cmd)
#     os.system(cmd)

    common_args = " --quiet --eval --skip_train --iteration 30000"
    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python render.py -m {out_base_path}/{out_name}/dtu_scan{scene} {common_args}'
    print(cmd)
    os.system(cmd)

    cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} python metrics.py " + \
          f"-m {out_base_path}/{out_name}/dtu_scan{scene} "
    print(cmd)
    os.system(cmd)

    break
