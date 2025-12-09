output_dir="/rsrch9/home/plm/idso_fa1_pathology/TIER2/sara-polyscope-tube"
script="polyscope-ai-tube/tube_processor.py"

# run on one svs example
input_file_path="/rsrch9/home/plm/idso_fa1_pathology/codes/sranjbar/polyscope_tube/test_one_case/MS001_HE.svs"
python3 $script --input_file_path "$input_file_path"  --output_dir "$output_dir"

# run on one tif example: you need to provide the objective power manually in this case
input_file_path="/rsrch9/home/plm/idso_fa1_pathology/TIER1/sara-tils-dcis/dcis_nki/pyramid_tifs/ptiff_test2/T14_68035_A7_HE_Default_Extended_p.tif"
python3 $script --input_file_path "$input_file_path"  --output_dir "$output_dir" --input_objective_power 40

# run in batch on directory
image_dir="/rsrch9/home/plm/idso_fa1_pathology/codes/sranjbar/polyscope_tube/anthrocosis_TMA5_examples"
file_pattern="*.svs"
find "$image_dir" -type f -name "$file_pattern" | while read -r img_path; do
    count=$((count + 1))
    echo "[$count / $total] Processing: $img_path"
    python3 $script --input_file_path "$img_path"  --output_dir "$output_dir"
done