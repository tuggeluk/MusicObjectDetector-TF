# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Convert deepscoresV2 dataset to TFRecord for object_detection.

Example usage:
    python object_detection/dataset_tools/create_deepscoresV2_tf_record.py \
        --data_dir=/home/user/DeepScoresV2 \
        --output_path=/home/user/deep_scoresV2.record
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import os

from lxml import etree
import PIL.Image
import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util
from tqdm import tqdm
import json

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to DeepScoresV2 dataset.')
flags.DEFINE_string('set', 'train', 'Convert train or val set')

flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS

SETS = ['train', 'val']


def dict_to_tf_example(data,
                       dataset_directory,
                       set_name,
                       id,
                       full_data):
    """Convert XML derived dict to tf.Example proto.

    Notice that this function normalizes the bounding box coordinates provided
    by the raw data.

    Args:
      data: dict holding PASCAL XML fields for a single image (obtained by
        running dataset_util.recursive_parse_xml_to_dict)
      dataset_directory: Path to root directory holding PASCAL dataset
      set_name: name of the set training, validation or test
      label_map_dict: A map from string label names to integers ids.
      ignore_difficult_instances: Whether to skip difficult instances in the
        dataset  (default: False).

    Returns:
      example: The converted tf.Example.

    Raises:
      ValueError: if the image pointed to by data['filename'] is not a valid JPEG
    """
    full_path = os.path.join(dataset_directory, 'images',data['filename'])
    with tf.gfile.GFile(full_path, 'rb') as fid:
        encoded_image = fid.read()
    encoded_image_io = io.BytesIO(encoded_image)
    image = PIL.Image.open(encoded_image_io)
    if image.format != 'JPEG' and image.format != 'PNG':
        raise ValueError('Image format not JPEG or PNG')
    key = hashlib.sha256(encoded_image).hexdigest()

    width = int(data['width'])
    height = int(data['height'])

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    truncated = []
    poses = []
    difficult_obj = []

    for obj in data['ann_ids']:

        ann = full_data['annotations'][obj]

        xmin.append(float(ann['a_bbox'][0]) / width)
        ymin.append(float(ann['a_bbox'][1]) / height)
        xmax.append(float(ann['a_bbox'][2]) / width)
        ymax.append(float(ann['a_bbox'][3]) / height)
        classes_text.append(full_data['categories'][ann['cat_id'][0]]['name'].encode('utf8'))
        classes.append(int(ann['cat_id'][0]))
        # Not sure if these are needed but ill leave them for compatibility
        truncated.append(int(0))
        poses.append("Unspecified".encode('utf8'))

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(
                data['filename'].encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(
                data['filename'].encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_image),
        'image/format': dataset_util.bytes_feature(image.format.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
        'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
        'image/object/truncated': dataset_util.int64_list_feature(truncated),
        'image/object/view': dataset_util.bytes_list_feature(poses),
    }))
    return example


def main(_):
    if FLAGS.set not in SETS:
        raise ValueError('set must be in : {}'.format(SETS))

    data_dir = FLAGS.data_dir

    os.makedirs(os.path.dirname(FLAGS.output_path), exist_ok=True)

    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

    with open(data_dir+"/deepscores_oriented_"+FLAGS.set+".json", 'r') as ann_file:
        data = json.load(ann_file)

    # maxi = 0
    # for _,ex in enumerate(data['images']):
    #     print(len(ex['ann_ids']))
    #     if len(ex['ann_ids']) > maxi:
    #         maxi = len(ex['ann_ids'])
    # print(maxi)
    for idx, example in tqdm(enumerate(data['images']),
                             desc="Parsing annotations from {0} set into TF-Example".format(FLAGS.set),
                             total=len(data['images'])):

        tf_example = dict_to_tf_example(example, FLAGS.data_dir, FLAGS.set, idx, data)
        writer.write(tf_example.SerializeToString())

    writer.close()

    # create mappings.txt file only on training
    if FLAGS.set == "train":
        with open(os.path.join(*FLAGS.output_path.split("/")[:-1]+["mappings_DeepScoresV2.txt"]), "w") as f:
            for k, v in data['categories'].items():
                f.write("""item {{
id: {0}
name: '{1}'
}}
""".format(int(k), v['name']))



if __name__ == '__main__':
    tf.app.run()
