$pathToGitRoot = "C:/Users/alpa/Repositories/MusicObjectDetector-TF/"
$pathToSourceRoot = "C:/Users/alpa/Repositories/MusicObjectDetector-TF/MusicObjectDetector/"
$pathToTranscript = "$($pathToSourceRoot)"

# Allowing wider outputs https://stackoverflow.com/questions/7158142/prevent-powergui-from-truncating-the-output
$pshost = get-host
$pswindow = $pshost.ui.rawui
$newsize = $pswindow.buffersize
$newsize.height = 9999
$newsize.width = 1500
$pswindow.buffersize = $newsize


cd C:/Users/alpa/Repositories/MusicObjectDetector-TF/research

Start-Transcript -path "$($pathToTranscript)Evaluate-ssd_inception_v2_muscima_150x300_pretrained.txt" -append
echo "Validate with ssd_inception_v2_muscima_150x300_pretrained"
python object_detection/eval.py --logtostderr --pipeline_config_path=C:\Users\alpa\Repositories\MusicObjectDetector-TF\MusicObjectDetector\configurations\ssd_inception_v2_muscima_150x300_pretrained.config --checkpoint_dir=C:\Users\alpa\Repositories\MusicObjectDetector-TF\MusicObjectDetector\data\training-checkpoints-ssd_inception_v2_muscima_150x300_pretrained --eval_dir=C:\Users\alpa\Repositories\MusicObjectDetector-TF\MusicObjectDetector\data\validation-checkpoints-ssd_inception_v2_muscima_150x300_pretrained
Stop-Transcript
