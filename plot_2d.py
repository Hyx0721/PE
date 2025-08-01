import numpy as np
import matplotlib.pyplot as plt
import os.path as osp
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
i="6"
model=["shallow","deep","EEG","conformer"]
save_name = f"Scene{i}-NP-EEGnet(B=32)" 
# save_name = f"Sub{i}-NP" 
pre = f" /test/BCML/memory_AAAI/new_model(token)/{save_name}_predicted_poses.txt"
tar = f" /test/BCML/memory_AAAI/new_model(token)/{save_name}_target_poses.txt"
# pre = f" /test/BCML/SAVE_AAAI_EEG/Sub/Shalow/{save_name}_predicted_poses.txt"
# tar = f" /test/BCML/SAVE_AAAI_EEG/Sub/Shalow/{save_name}_target_poses.txt"
# /test/BCML/memory_AAAI/new_model(token)/Sub1-ica-EEGnet(B=32)-abaltion(cross)-3_predicted_poses.txt
# /test/BCML/SAVE_AAAI_EEG/Sub/conformer/Sub1-NP-conformer.txt
save_dir = " /test/BCML/Pic_AAAI/"
# /test/BCML/memory_AAAI/new_model(token)/Scene1-NP-EEGnet(B=32)_predicted_poses.txt
# pre = f" /test/BCML/memory_EEG/{save_name}_predicted_poses.txt"
# tar = f" /test/BCML/memory_EEG/{save_name}_target_poses.txt"



image_filename = osp.join(save_dir, f"{save_name}-2d.png")

gt_pose = np.loadtxt(tar)         # shape: [N, 7] - GT
pred_pose = np.loadtxt(pre)       # shape: [N, 7] - Predicted
assert gt_pose.shape[0] == pred_pose.shape[0], "预测和GT行数不一致"
assert gt_pose.shape[1] >= 3, "位置列数不足"

gt_xyz = gt_pose[:, :3]
pred_xyz = pred_pose[:, :3]

fig1, ax1 = plt.subplots(figsize=(12, 9.6))
ax1.scatter(gt_pose[:, 0], gt_pose[:, 1],color='black', s=80,marker='o', linestyle='',)
ax1.scatter(pred_pose[:, 0], pred_pose[:, 1], color='blue', s=80,marker='s', linestyle='')

for i in range(len(gt_pose)):
    ax1.plot([gt_pose[i, 0], pred_pose[i, 0]], [gt_pose[i, 1], pred_pose[i, 1]],color='red', linestyle='-', linewidth=2, alpha=0.5)

ax1.set_xlabel('X [m]', fontsize=30, labelpad=5)
ax1.set_ylabel('Y [m]', fontsize=30, labelpad=-10)
ax1.xaxis.set_major_locator(MaxNLocator(nbins=5))
ax1.yaxis.set_major_locator(MaxNLocator(nbins=5))
ax1.tick_params(axis='both', labelsize=35, width=1)
ax1.grid()




fig1.savefig(image_filename)
plt.close()

# Print output
print(f"3D plot saved to: {image_filename}")