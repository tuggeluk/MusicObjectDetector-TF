$pathToGitRoot = "C:/Users/Alex/Repositories/MusicObjectDetector-TF2"
$pathToSourceRoot = "$($pathToGitRoot)/MusicObjectDetector"
$pathToTranscript = "$($pathToSourceRoot)/Transcripts"
$pathToData = "$($pathToSourceRoot)/data"
#$pathToData = "\\MONSTI\MusicObjectDetector-TF_Results"
cd $pathToGitRoot

#echo "Appending required paths to temporary PYTHONPATH"
#$env:PYTHONPATH = "$($pathToGitRoot);$($pathToGitRoot)/research;$($pathToSourceRoot)"



################################################################
# Available configurations - uncomment the one to actually run #
################################################################
#$configuration = "faster_rcnn_inception_resnet_v2_atrous_pretrained_dimension_clustering_rms"
#$configuration = "faster_rcnn_inception_resnet_v2_atrous_pretrained_dimension_clustering_rms_1200_proposals"
#$configuration = "faster_rcnn_inception_resnet_v2_atrous_pretrained_dimension_clustering_rms_2000_proposals"
#$configuration = "faster_rcnn_inception_resnet_v2_atrous_pretrained_deepscores_1"
#$configuration = "faster_rcnn_inception_resnet_v2_atrous_pretrained_mensural_1"
#$configuration = "faster_rcnn_inception_resnet_v2_atrous_pretrained_muscima_1"
#$configuration = "faster_rcnn_inc_resnet_v2_muscima_1"
$configuration = "ssd_resnet50_retinanet_muscima_1"

echo "Training with $($configuration) configuration"

# Legacy slim-based
Start-Transcript -path "$($pathToTranscript)/Train-$($configuration).txt" -append
python research/object_detection/legacy/train.py --pipeline_config_path="$($pathToSourceRoot)/configurations/$($configuration).config" --train_dir="$($pathToData)/checkpoints-$($configuration)-train"
Stop-Transcript

# # Estimator-based
# Start-Transcript -path "$($pathToTranscript)/TrainEval-$($configuration).txt" -append
# python research/object_detection/model_main.py --alsologtostderr --pipeline_config_path="$($pathToSourceRoot)/configurations/$($configuration).config" --model_dir="$($pathToData)/$($configuration)"
# Stop-Transcript
