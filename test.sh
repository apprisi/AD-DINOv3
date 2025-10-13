declare -a dataset=(mvtec visa btad mpdd tn3k clinicdb colondb isic)
save_path="./TESTING_ALL"
for i in "${dataset[@]}"; do
    python test.py --result_path $save_path --dataset $i
done