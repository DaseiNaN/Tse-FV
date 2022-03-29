for f in $(find . -name "*.flac"); do
    ffmpeg-normalize "$f" -ar 16000 -o "${f%.*}-norm.wav"
done