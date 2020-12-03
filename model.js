// import * as tf from '@tensorflow/tfjs';
// import * as tfd from '@tensorflow/tfjs-data';
// import * as blazeface from '@tensorflow-models/blazeface';

let video, videoWidth, videoHeight;
async function setupCamera() {
  video = document.getElementById('video');

  const stream = await navigator.mediaDevices.getUserMedia({
    'audio': false,
    'video': { facingMode: 'user' },
  });
  video.srcObject = stream;

  return new Promise((resolve) => {
    video.onloadedmetadata = () => {
      resolve(video);
    };
  });
}

let canvas, canvasCtx;
async function setupCanvas() {
  canvas = document.getElementById('output');
  canvas.width = videoWidth;
  canvas.height = videoHeight;
  canvasCtx = canvas.getContext('2d');
  canvasCtx.fillStyle = "rgba(255, 0, 0, 0.5)";
}

let faceDetectionModel;
async function loadFaceDetectionModel() {
  console.log("loading face detection model");
  await blazeface.load().then(m => {
    faceDetectionModel = m;
    console.log("face detection model loaded");
  });
}

let maskDetectionModel;
async function loadMaskDetectionModel() {
  console.log("loading mask detection model");
  await tf.loadLayersModel('modeljs/model.json').then(m => {
    maskDetectionModel = m;
    console.log("mask detection model loaded");
  });
}

const returnTensors = false;
const flipHorizontal = true;
const annotateBoxes = false;
const offset = tf.scalar(127.5);
const mask_labels = {0: 'without_mask', 1: 'with_mask', 2: 'mask_weared_incorrect'};


const loadingModel = document.getElementById('loading-model');

function indexOfMax(arr) {
    if (arr.length === 0) {
        return -1;
    }

    var max = arr[0];
    var maxIndex = 0;

    for (var i = 1; i < arr.length; i++) {
        if (arr[i] > max) {
            maxIndex = i;
            max = arr[i];
        }
    }

    return maxIndex;
}

async function renderPrediction() {
  // Get image from webcam
  let img = tf.tidy(() => tf.browser.fromPixels(video));

  // Detect faces
  let faces = [];
  try {
    faces = await faceDetectionModel.estimateFaces(img, annotateBoxes,flipHorizontal, returnTensors);
  } catch (e) {
    console.error("estimateFaces:", e);
    return;
  }

  if (faces.length > 0) {
    // TODO: Loop through all predicted faces and detect if mask used or not.
    // RIght now, it only highlights the first face into the live view. (See the break command below)
    for (let i = 0; i < faces.length; i++) {
      let predictions = [];

      let face = tf.tidy(() => img.resizeNearestNeighbor([150, 150])
        .toFloat().sub(offset).div(offset).expandDims(0));

      try {
        predictions = await maskDetectionModel.predict(face).data();
      } catch (e){
        console.error("maskDetection:", e);
        return;
      }

      face.dispose();

      const start = faces[i].topLeft;
      const end = faces[i].bottomRight;
      const size = [end[0] - start[0], end[1] - start[1]];
      const colors = {0: "255, 0, 0", 1: "0, 255, 0", 2: "255, 255, 0"} // set colors

      canvasCtx.clearRect(0, 0, canvas.width, canvas.height);

      if (predictions.length > 0) {
          let target = indexOfMax(predictions);
          label = `${mask_labels[target]}: ${Math.floor(predictions[target] * 1000) / 10}%`;

        // Render label and its box
        canvasCtx.fillStyle = "rgba("+colors[target]+", 0.85)";
        canvasCtx.fillRect(start[0], start[1] - 23, size[0], 23);
        canvasCtx.font = "18px Raleway";
        canvasCtx.fillStyle = "rgba(255, 255, 255, 1)";
        canvasCtx.fillText(label, end[0] + 5, start[1] - 5);
      }

      

      // TODO: Loop through all detected faces instead of the first one.
      break;
    }
  }

  img.dispose();
  requestAnimationFrame(renderPrediction);

  if (loadingModel.innerHTML !== "") {
    loadingModel.innerHTML = "";
  }
}

async function main() {
  await setupCamera();
  video.play();

  videoWidth = video.videoWidth;
  videoHeight = video.videoHeight;
  video.width = videoWidth;
  video.height = videoHeight;

  setupCanvas();

  await loadFaceDetectionModel();
  await loadMaskDetectionModel();

  renderPrediction();
}

main();