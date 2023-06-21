# altin-thermal


## FFmpeg cam capture and display
```bash
# Base streams
ffplay -f avfoundation -framerate 25 -video_size 256x384 -pixel_format yuyv422 -i "0"
ffplay -f avfoundation -framerate 25 -video_size 256x192 -pixel_format yuyv422 -i "0"

# False color gradient
ffmpeg -f avfoundation -framerate 25 -pixel_format yuyv422 -video_size 256x384 -i "0" -vf "crop=h=(ih/2):y=(ih/2)" -pix_fmt yuyv422 -f rawvideo - | ffplay -pixel_format gray16le -video_size 256x192 -f rawvideo - -vf 'normalize=smoothing=10, format=pix_fmts=rgb48, pseudocolor=p=inferno'

# With computed metrics
ffmpeg -f avfoundation -framerate 25 -pixel_format yuyv422 -video_size 256x384 -i "0" -vf "crop=h=(ih/2):y=(ih/2)" -pix_fmt yuyv422 -f rawvideo - | ffplay -pixel_format gray16le -video_size 256x192 -f rawvideo - -vf 'signalstats, split [main][secondary]; [main] normalize=smoothing=10, format=pix_fmts=rgb48, pseudocolor=p=inferno, scale=w=2*iw:h=2*ih, drawtext=x=3:y=3:borderw=1:bordercolor=white:fontfile=FreeSerif.ttf:text=MIN\\: %{metadata\\:lavfi.signalstats.YMIN}    MAX\\: %{metadata\\:lavfi.signalstats.YMAX} [thermal]; [secondary] drawgraph=m1=lavfi.signalstats.YMIN:fg1=0xFFFF9040:m2=lavfi.signalstats.YMAX:fg2=0xFF0000FF:bg=0x303030:min=18500:max=24500:slide=scroll:size=512x64 [graph]; [thermal][graph] vstack'
```