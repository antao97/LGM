''' Analysis script

Dynamics-aware Adversarial Attack of Adaptive Neural Networks

Author: Yingqi Wang, An Tao
Email: yingqi-w19@mails.tsinghua.edu.cn, ta19@mails.tsinghua.edu.cn
Date: 2023/10/31

'''

import os
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import cv2
from sklearn.decomposition import PCA
import sklearn.manifold as manifold

num_class = 1000
num_layer = 32
classes = [[7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146], [151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277], [101, 102, 103, 104, 105, 106, 148, 149, 150, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388], [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 111], [0, 1, 2, 3, 4, 5, 6, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 107, 108, 109, 110, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 147, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 389, 390, 391, 392, 393, 394, 395, 396, 397], [509, 599, 692, 907, 923, 924, 925, 926, 927, 928, 929, 930, 931, 932, 933, 934, 935, 936, 937, 938, 939, 940, 941, 942, 943, 944, 945, 947, 948, 949, 950, 951, 952, 953, 954, 955, 956, 957, 959, 960, 961, 962, 963, 964, 965, 966, 967, 968, 969, 984, 986, 987, 988, 989, 990, 991, 992, 993, 994, 995, 996, 997, 998], [401, 402, 409, 418, 419, 420, 422, 423, 426, 429, 430, 432, 446, 447, 451, 455, 457, 459, 462, 470, 473, 475, 479, 481, 482, 485, 486, 487, 488, 494, 507, 508, 513, 522, 523, 526, 527, 528, 529, 530, 531, 532, 533, 535, 539, 541, 542, 543, 544, 545, 546, 548, 549, 550, 551, 552, 558, 559, 560, 563, 564, 566, 567, 568, 570, 574, 577, 579, 584, 585, 587, 589, 591, 592, 593, 594, 598, 604, 605, 606, 607, 611, 619, 620, 621, 622, 623, 626, 629, 631, 632, 633, 635, 641, 642, 643, 644, 645, 646, 650, 662, 664, 673, 676, 677, 681, 683, 684, 687, 695, 696, 699, 700, 710, 711, 713, 714, 721, 722, 726, 731, 732, 733, 736, 738, 741, 742, 745, 746, 747, 749, 750, 752, 753, 754, 755, 758, 759, 761, 767, 768, 769, 772, 776, 778, 782, 783, 784, 785, 786, 789, 792, 793, 794, 796, 797, 798, 804, 805, 838, 850, 851, 852, 890, 946, 981, 999], [410, 412, 414, 415, 431, 435, 438, 440, 441, 448, 453, 463, 469, 478, 492, 493, 495, 503, 504, 505, 516, 519, 520, 521, 534, 553, 572, 588, 618, 636, 637, 647, 648, 651, 653, 659, 666, 680, 709, 712, 719, 720, 725, 728, 729, 737, 748, 756, 760, 766, 771, 773, 790, 791, 809, 849, 855, 861, 868, 876, 883, 891, 893, 894, 896, 897, 898, 899, 900, 901, 909, 910], [398, 400, 411, 427, 433, 434, 439, 443, 445, 452, 456, 461, 464, 465, 471, 474, 477, 491, 496, 499, 501, 502, 512, 514, 515, 517, 518, 524, 578, 596, 597, 601, 608, 610, 614, 615, 617, 630, 638, 639, 655, 658, 667, 678, 679, 688, 689, 693, 697, 701, 715, 735, 770, 774, 775, 787, 801, 806, 808, 813, 818, 823, 834, 836, 837, 840, 841, 842, 869, 879, 887, 906, 911], [406, 416, 421, 424, 425, 437, 442, 449, 454, 458, 467, 476, 480, 483, 489, 490, 497, 498, 500, 525, 536, 538, 540, 554, 556, 557, 562, 571, 580, 582, 583, 624, 649, 663, 668, 669, 672, 682, 698, 706, 708, 716, 718, 723, 727, 743, 762, 788, 799, 821, 827, 832, 839, 843, 853, 860, 865, 872, 873, 884, 888, 912, 915, 958], [403, 404, 405, 407, 408, 417, 428, 436, 444, 450, 466, 468, 472, 484, 510, 511, 537, 547, 555, 561, 565, 569, 573, 575, 576, 581, 586, 595, 603, 609, 612, 625, 627, 628, 654, 656, 660, 661, 665, 670, 671, 675, 690, 705, 717, 724, 730, 734, 751, 757, 765, 779, 780, 802, 803, 812, 814, 817, 820, 824, 829, 830, 833, 847, 856, 864, 866, 867, 870, 871, 874, 880, 895, 908, 914, 919, 920], [413, 506, 590, 600, 602, 613, 616, 634, 640, 657, 674, 685, 686, 691, 694, 702, 703, 704, 707, 739, 740, 744, 763, 764, 777, 781, 800, 807, 810, 811, 816, 822, 826, 828, 831, 835, 844, 845, 846, 848, 854, 858, 859, 862, 863, 875, 877, 878, 881, 882, 886, 889, 892, 902, 904, 905, 913, 916, 917, 918, 921], [399, 460, 652, 795, 815, 819, 825, 857, 885, 903, 922, 970, 971, 972, 973, 974, 975, 976, 977, 978, 979, 980, 982, 983, 985]]
classnames = [
    'birds',
    'dogs',
    'other mammals',
    'reptiles',
    'other animals',
    'food',
    'cosumer goods',
    'container',
    'equipment',
    'structures',
    'transport',
    'other objects',
    'other',
]
s = []
for i in classes:
    for j in i:
        s.append(j)

classes_length = [59, 127, 90, 45, 77, 63, 167, 72, 73, 64, 77, 61, 25]

exp_name = 'eps16_iter100_lr1_ce_mask1.0'

def get_tsne(tsne, input):
    input_tsne = tsne.fit_transform(input)
    input_tsne_min, input_tsne_max = input_tsne.min(0), input_tsne.max(0)
    input_tsne = (input_tsne - input_tsne_min) / (input_tsne_max - input_tsne_min)
    return input_tsne

def draw():
    fgm_path = os.path.join('outputs/skipnet/imagenet/fgm', exp_name)
    lgm_path = os.path.join('outputs/skipnet/imagenet/lgm', exp_name)
    save_path = os.path.join('analysis/skipnet/imagenet', exp_name)
    os.makedirs(save_path, exist_ok=True)

    activate_matrix_origin_fgm = np.zeros((num_class, num_layer))
    activate_matrix_with_true_label_fgm = np.zeros((num_class, num_layer))
    activate_matrix_with_wrong_label_fgm = np.zeros((num_class, num_layer))
    all_matrix_fgm = np.zeros((num_class, num_layer))
    all_matrix_wrong_fgm = np.zeros((num_class, num_layer))

    fgm_file_path = os.path.join(fgm_path, 'out.pickle')
    with open(fgm_file_path, 'rb') as f:
        while(1):
            try:
                data = pickle.load(f)
                activate_matrix_origin_fgm[data['gt']] += data['mask_o'].astype(int)
                activate_matrix_with_true_label_fgm[data['gt']] += data['mask'].astype(int)
                prec_idx = np.argmax(data['prec'])
                activate_matrix_with_wrong_label_fgm[prec_idx] += data['mask'].astype(int)
                all_matrix_fgm[data['gt']] += 1
                all_matrix_wrong_fgm[prec_idx] += 1
                
            except EOFError:
                break

    hm1 = activate_matrix_origin_fgm/(all_matrix_fgm+1e-8)
    hm2 = activate_matrix_with_true_label_fgm/(all_matrix_fgm+1e-8)
    hm3 = activate_matrix_with_wrong_label_fgm/(all_matrix_wrong_fgm+1e-8)

    
    activate_matrix_origin_lgm = np.zeros((num_class, num_layer))
    activate_matrix_with_true_label_lgm = np.zeros((num_class, num_layer))
    activate_matrix_with_wrong_label_lgm = np.zeros((num_class, num_layer))
    all_matrix_lgm = np.zeros((num_class, num_layer))
    all_matrix_wrong_lgm = np.zeros((num_class, num_layer))

    lgm_file_path = os.path.join(lgm_path, 'out.pickle')
    with open(lgm_file_path, 'rb') as f:
        while(1):
            try:
                data = pickle.load(f)
                activate_matrix_origin_lgm[data['gt']] += data['mask_o'].astype(int)
                activate_matrix_with_true_label_lgm[data['gt']] += data['mask'].astype(int)
                prec_idx = np.argmax(data['prec'])
                activate_matrix_with_wrong_label_lgm[prec_idx] += data['mask'].astype(int)
                all_matrix_lgm[data['gt']] += 1
                all_matrix_wrong_lgm[prec_idx] += 1
                
            except EOFError:
                break

    hm4 = activate_matrix_with_true_label_lgm/(all_matrix_lgm+1e-8)
    hm5 = activate_matrix_with_wrong_label_lgm/(all_matrix_wrong_lgm+1e-8)

    fig, axes = plt.subplots(1, 6, figsize=(18, 3), sharex=True)
    plt.subplots_adjust(wspace=0.95)

    # Draw heatmap
    sns.heatmap(hm1[s], annot=False, cmap='YlGnBu', vmax=1, vmin=0, cbar=False, ax = axes[0], yticklabels=False)
    sns.heatmap(hm2[s], annot=False, cmap='YlGnBu', vmax=1, vmin=0, cbar=False, ax = axes[1], yticklabels=False)
    sns.heatmap(hm3[s], annot=False, cmap='YlGnBu', vmax=1, vmin=0, cbar=False, ax = axes[2], yticklabels=False)
    sns.heatmap(hm4[s], annot=False, cmap='YlGnBu', vmax=1, vmin=0, cbar=False, ax = axes[3], yticklabels=False)
    sns.heatmap(hm5[s], annot=False, cmap='YlGnBu', vmax=1, vmin=0, cbar=False, ax = axes[4], yticklabels=False)
    axes[5].axis('off')
    plt.colorbar(ax=axes[5], mappable=matplotlib.cm.ScalarMappable(cmap='YlGnBu'))
    
    fig.savefig(os.path.join(save_path, 'heatmap0.jpg'), dpi=1000)
    plt.close()

    # Add class label
    img = cv2.imread(os.path.join(save_path, 'heatmap0.jpg'))
    for fig in range(5):
        topx = 2250 + fig * 2530
        topy = 360
        lowy = 2670
        l = 60
        w = 10

        y = [0]
        for i in classes_length:
            y.append(y[-1]+i)
        y = np.asarray(y)
        y = y/y[-1]*(lowy-topy-w)+topy+w/2
        y = y.astype(int)

        for i in y:
            cv2.line(img, (topx-l, i), (topx, i), (0,0,0), w)

        offsets = [432, 442, -260, 285, -130, 465, -230, 145, 80, 110, 155, -100, 400]
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i, classname in enumerate(classnames):
            if i == 12:
                cv2.putText(img, classname, (1400+fig*2530+offsets[i], int((y[i]+y[i+1])/2)+50), font, 4, (0,0,0), 10)
            else:
                cv2.putText(img, classname, (1400+fig*2530+offsets[i], int((y[i]+y[i+1])/2)+20), font, 4, (0,0,0), 10)

    cv2.imwrite(os.path.join(save_path, 'heatmap.jpg'), img)

    # PCA
    pca_func = PCA(n_components=2)
    pca1 = pca_func.fit(hm1).transform(hm1)
    pca2 = pca_func.fit(hm2).transform(hm2)
    pca3 = np.zeros(pca1.shape)
    mask3 = np.ones(hm3.shape[0], dtype=bool)
    mask3[hm3.sum(1)==0] = False
    pca3[mask3] = pca_func.fit(hm3[mask3]).transform(hm3[mask3])
    pca4 = pca_func.fit(hm4).transform(hm4)
    pca5 = np.zeros(pca1.shape)
    mask5 = np.ones(hm5.shape[0], dtype=bool)
    mask5[hm5.sum(1)==0] = False
    pca5[mask5] = pca_func.fit(hm5[mask5]).transform(hm5[mask5])
    color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#ba2be2', '#006400', '#7b68ee']
    fig, axes = plt.subplots(1, 6, figsize=(18, 3), sharex=True, sharey=True)
    ax = []
    for i, j in enumerate(classes):
        mask = np.ones(len(j), dtype=bool)
        mask[hm3[j].sum(1)==0] = False
        mask[hm5[j].sum(1)==0] = False
        j = np.array(j)[mask]
        axes[0].scatter(pca1[j,0], pca1[j,1], alpha=.8, lw=2, s=1, label=classnames[i], c=color[i])
        axes[1].scatter(pca2[j,0], pca2[j,1], alpha=.8, lw=2, s=1, label=classnames[i], c=color[i])
        axes[2].scatter(pca3[j,0], pca3[j,1], alpha=.8, lw=2, s=1, label=classnames[i], c=color[i])
        axes[3].scatter(pca4[j,0], pca4[j,1], alpha=.8, lw=2, s=1, label=classnames[i], c=color[i])
        ax.append(axes[4].scatter(pca5[j,0], pca5[j,1], alpha=.8, lw=2, s=1, label=classnames[i], c=color[i]))
    
    axes[5].axis('off')
    axes[5].legend(ax, classnames, fontsize=8)

    fig.savefig(os.path.join(save_path, 'pca.jpg'), dpi=1000)
    plt.close()


    # T-SNE
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    hm1_tsne = get_tsne(tsne, hm1)
    hm2_tsne = get_tsne(tsne, hm2)
    hm3_tsne = np.zeros(hm1_tsne.shape)
    mask3 = np.ones(hm3.shape[0], dtype=bool)
    mask3[hm3.sum(1)==0] = False
    hm3_tsne[mask3] = get_tsne(tsne, hm3[mask3])
    hm4_tsne = get_tsne(tsne, hm4)
    hm5_tsne = np.zeros(hm1_tsne.shape)
    mask5 = np.ones(hm5.shape[0], dtype=bool)
    mask5[hm5.sum(1)==0] = False
    hm5_tsne[mask5] = get_tsne(tsne, hm5[mask5])
    fig, axes = plt.subplots(1, 6, figsize=(18, 3), sharex=True, sharey=True)
    ax = []
    for i, j in enumerate(classes):
        mask = np.ones(len(j), dtype=bool)
        mask[hm3[j].sum(1)==0] = False
        mask[hm5[j].sum(1)==0] = False
        j = np.array(j)[mask]
        axes[0].scatter(hm1_tsne[j,0], hm1_tsne[j,1], alpha=.8, lw=2, s=1, label=classnames[i], c=color[i])
        axes[1].scatter(hm2_tsne[j,0], hm2_tsne[j,1], alpha=.8, lw=2, s=1, label=classnames[i], c=color[i])
        axes[2].scatter(hm3_tsne[j,0], hm3_tsne[j,1], alpha=.8, lw=2, s=1, label=classnames[i], c=color[i])
        axes[3].scatter(hm4_tsne[j,0], hm4_tsne[j,1], alpha=.8, lw=2, s=1, label=classnames[i], c=color[i])
        ax.append(axes[4].scatter(hm5_tsne[j,0], hm5_tsne[j,1], alpha=.8, lw=2, s=1, label=classnames[i], c=color[i]))
    
    axes[5].axis('off')
    axes[5].legend(ax, classnames, fontsize=8)

    fig.savefig(os.path.join(save_path, 't-sne.jpg'), dpi=1000)
    plt.close()
    
draw()
