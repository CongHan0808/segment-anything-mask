# python scripts/amg.py --checkpoint "segment_anything/pretrain_weights/sam_vit_b_01ec64.pth" \
python scripts/amg.py --checkpoint "segment_anything/pretrain_weights/sam_vit_h_4b8939.pth" \
--model-type "vit_h" \
--input "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/hancong/code/pretrain/ov-seg/zsseg.baseline/job/000000000872.jpg" \
--output "test/vith"
