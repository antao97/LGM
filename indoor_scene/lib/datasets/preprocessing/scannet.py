from pathlib import Path

import numpy as np
from lib.pc_utils import read_plyfile, save_point_cloud
from concurrent.futures import ProcessPoolExecutor

import os, csv, json

SCANNET_RAW_PATH = Path('/data3/antao/Documents/Datasets/ScanNet_raw')
SCANNET_OUT_PATH = Path('/data3/antao/Documents/Datasets/ScanNet_processed/')
TRAIN_DEST = 'train'
TEST_DEST = 'test'
SUBSETS = {TRAIN_DEST: 'scans', TEST_DEST: 'scans_test'}
POINTCLOUD_FILE = '_vh_clean_2.ply'
BUGS = {
    'train/scene0270_00.ply': 50,
    'train/scene0270_02.ply': 50,
    'train/scene0384_00.ply': 149,
}
print('start preprocess')
# Preprocess data.

def load_labels(label_path):
  if label_path.endswith('.txt'):
    with open(label_path, 'r') as f:
      file = f.readlines()
    f.close()
    labels = []
    for i in range(len(file)):
      labels.append(int(file[i][:-1]))
  elif label_path.endswith('.json'):
    with open(label_path, 'r') as f:
      file = json.load(f)
    f.close()
    labels = file["segIndices"]
  else:
    print('Not supported file type!')
    exit(1)
  return labels


def load_seg_labels(label_file):
  with open(label_file, 'r') as f:
    file = json.load(f)
  f.close()
  labels = file["segIndices"]
  return labels


def read_label_mapper(filename, label_from='raw_category', label_to='nyu40id'):
  assert os.path.isfile(filename)
  mapper = dict()
  with open(filename) as csvfile:
    reader = csv.DictReader(csvfile, delimiter='\t')
    for row in reader:
      mapper[row[label_from]] = int(row[label_to])
  return mapper


def load_aggregation(aggregation_file, mapper):
  with open(aggregation_file, 'r') as f:
    file = json.load(f)
  f.close()
  seg2ins = {}
  seg2sem = {}
  for seg in file["segGroups"]:
    if (aggregation_file.split('/')[-1][:12] == 'scene0217_00') and (seg['objectId'] == 31):
      break
    for seg_label in seg['segments']:
      seg2ins.update({seg_label: seg['objectId']+1})
      seg2sem.update({seg_label: mapper[seg['label']]})
  return seg2ins, seg2sem


def load_labels(scene_path):
  scene_name = scene_path.stem
  seg_path = scene_path / (scene_name + '_vh_clean_2.0.010000.segs.json')
  aggregation_path = scene_path / (scene_name + '.aggregation.json')
  mapper_path = scene_path.parent.parent / 'scannetv2-labels.combined.tsv'
  seg_labels = np.array(load_seg_labels(seg_path))

  mapper = read_label_mapper(mapper_path)
  seg2ins, seg2sem = load_aggregation(str(aggregation_path), mapper)

  ins_labels = np.zeros(seg_labels.shape)
  sem_labels = np.zeros(seg_labels.shape)
  for seg_unique in np.unique(seg_labels):
    indexs = np.where(seg_labels==seg_unique)[0]
    if seg_unique in seg2ins.keys():
      ins = seg2ins[seg_unique]
      sem = seg2sem[seg_unique]
    else:
      ins = 0
      sem = 0
    ins_labels[indexs] = ins
    sem_labels[indexs] = sem
  return ins_labels, sem_labels


def handle_process(path):
  f = Path(path.split(',')[0])
  phase_out_path = Path(path.split(',')[1])
  pointcloud = read_plyfile(f)
  # Make sure alpha value is meaningless.
  assert np.unique(pointcloud[:, -1]).size == 1
  # Load label file.
  label_f = f.parent / (f.stem + '.labels' + f.suffix)
  if label_f.is_file():
    # label = read_plyfile(label_f)
    ins_labels, sem_labels = load_labels(f.parent)
    # Sanity check that the pointcloud and its label has same vertices.
    # assert pointcloud.shape[0] == label.shape[0]
    # assert np.allclose(pointcloud[:, :3], label[:, :3])
  else:  # Label may not exist in test case.
    ins_labels = np.zeros_like(pointcloud)[:, -1]
    sem_labels = np.zeros_like(pointcloud)[:, -1]
  
  
  out_f = phase_out_path / (f.name[:-len(POINTCLOUD_FILE)] + f.suffix)
  # processed = np.hstack((pointcloud[:, :6], np.array([label[:, -1]]).T))
  processed = np.hstack((pointcloud[:, :6], np.array([ins_labels]).T, np.array([sem_labels]).T))
  save_point_cloud(processed, out_f, with_label=True, verbose=False)


path_list = []
for out_path, in_path in SUBSETS.items():
  phase_out_path = SCANNET_OUT_PATH / out_path
  phase_out_path.mkdir(parents=True, exist_ok=True)
  for f in (SCANNET_RAW_PATH / in_path).glob('*/*' + POINTCLOUD_FILE):
    path_list.append(str(f) + ',' + str(phase_out_path))

print(len(path_list))
pool = ProcessPoolExecutor(max_workers=20)
result = list(pool.map(handle_process, path_list))
# for path in path_list:
#   handle_process(path)

# Fix bug in the data.
for files, bug_index in BUGS.items():
  print(files)

  for f in SCANNET_OUT_PATH.glob(files):
    pointcloud = read_plyfile(f)
    bug_mask = pointcloud[:, -1] == bug_index
    print(f'Fixing {f} bugged label {bug_index} x {bug_mask.sum()}')
    pointcloud[bug_mask, -1] = 0
    save_point_cloud(pointcloud, f, with_label=True, verbose=False)
