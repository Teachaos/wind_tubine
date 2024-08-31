document.addEventListener('DOMContentLoaded', function() {
    const video = document.getElementById('video');
    const playButton = document.getElementById('play');
    const pauseButton = document.getElementById('pause');
    const imageCanvas = document.getElementById('imageCanvas');
    const ctx = imageCanvas.getContext('2d');

    // 添加一个变量来跟踪是否已经设置了画布尺寸
    let canvasResized = false;

    // 监听视频的 canplay 事件，以设置 canvas 的尺寸
    video.addEventListener('canplay', function() {
        if (!canvasResized) {
            imageCanvas.width = video.videoWidth;
            imageCanvas.height = video.videoHeight;
            canvasResized = true;
        }
    });

    // 视频播放控制
    playButton.addEventListener('click', function() {
        video.play();
    });

    pauseButton.addEventListener('click', function() {
        video.pause();
    });

    // 处理图片
    function processImage() {
        // 只在视频正在播放时才绘制
        if (video.paused || video.ended) return;

        // 将视频帧绘制到 canvas 上
        ctx.drawImage(video, 0, 0, imageCanvas.width, imageCanvas.height);

        // 图像处理 - 边缘锐化
        const imageData = ctx.getImageData(0, 0, imageCanvas.width, imageCanvas.height);
        const data = imageData.data;

        // 简单的锐化滤镜
        const kernel = [
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ];

        // 对每个像素进行卷积计算
        for (let y = 1; y < imageCanvas.height - 1; y++) {
            for (let x = 1; x < imageCanvas.width - 1; x++) {
                let r = 0, g = 0, b = 0;
                for (let ky = -1; ky <= 1; ky++) {
                    for (let kx = -1; kx <= 1; kx++) {
                        const idx = ((y + ky) * imageCanvas.width + (x + kx)) * 4;
                        const kidx = (ky + 1) * 3 + (kx + 1);
                        r += data[idx] * kernel[kidx];
                        g += data[idx + 1] * kernel[kidx];
                        b += data[idx + 2] * kernel[kidx];
                    }
                }

                const idx = (y * imageCanvas.width + x) * 4;
                data[idx] = Math.min(255, Math.max(0, r));
                data[idx + 1] = Math.min(255, Math.max(0, g));
                data[idx + 2] = Math.min(255, Math.max(0, b));
            }
        }

        ctx.putImageData(imageData, 0, 0);
        ctx.drawImage(video, 0, 0, imageCanvas.width, imageCanvas.height);
    }

    // 当视频播放时，每秒处理一次图像
    video.addEventListener('timeupdate', function() {
        processImage();
    });
});