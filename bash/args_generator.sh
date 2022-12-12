epoch=100
num_actions=100000
training_filename="train.py"
output_filename="args.txt"
hidden_layer=64
batch_size=64
lr=1e-4
env_ids=("Ant-v3" "HalfCheetah-v3" "Hopper-v3" "Walker2d-v3" "Swimmer20-v3" "ReacherTracker20-v3")
native_dims=(8 6 3 6 19 20)
# env_ids=("Swimmer6-v3" "Swimmer10-v3" "Swimmer20-v3" "ReacherTracker10-v3" "ReacherTracker20-v3")
# native_dims=(5 9 19 10 20)

for i in {0..5}
do
    env_id=${env_ids[${i}]}
    native_dim=${native_dims[${i}]}
    num_records=1
    
    for ((latent_dim=1; latent_dim<=${native_dim}; latent_dim++))
    do  
        model='AE'
        save_dir='./random'
        echo "${training_filename}, ${env_id}, ${native_dim}, ${latent_dim}, ${model}, ${hidden_layer}, ${batch_size}, ./${env_id}/random_actions.npy, ${save_dir}, ${epoch}, ${lr}, ${num_actions}, ${num_records}" >> ${output_filename}

    done

    num_records=2
    model='OTNAE'
    save_dir='./random'
    echo "${training_filename}, ${env_id}, ${native_dim}, ${native_dim}, ${model}, ${hidden_layer}, ${batch_size}, ./${env_id}/random_actions.npy, ${save_dir}, ${epoch}, ${lr}, ${num_actions}, ${num_records}" >> ${output_filename}

done
