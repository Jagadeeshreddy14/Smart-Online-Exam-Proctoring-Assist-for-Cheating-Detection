
var startButton = document.querySelector("#startQuiz");     //Main page start button
var timer = document.querySelector("#timer");   //Timer when quiz starts
var mainContent = document.querySelector("#mainContent");   //Start page content div
var questionEl = document.querySelector("#title");  //card title
var quizContent = document.querySelector("#quizContent");   //Question options div
var resultDiv = document.querySelector("#answer");  //Div for showing answer is correct/wrong
var completeTest = document.querySelector("#completeTest");    //Div for Displying final scores when quiz completed
var highscoresDiv = document.querySelector("#highscores");  //Div for showing highscores
var navhighscorelink = document.querySelector("#navhighscorelink");     //highscore navigation link
var navlink = document.getElementById("navhighscorelink");

var secondsLeft = 300, questionIndex = 0, correct = 0;
var totalQuestions = questions.length;
var question, option1, option2, option3, option4, ans, previousScores;
var choiceArray = [], divArray = [];
var mediaRecorder;
var recordedChunks = [];
var userLocation = { latitude: null, longitude: null };
var examSessionId = "session_" + Date.now();

// Toast notification function
function showToast(message, type = 'success') {
    const container = document.getElementById('toastContainer');
    if (!container) return;

    const toast = document.createElement('div');
    const bgColor = type === 'success' ? 'linear-gradient(135deg, #10b981, #059669)' :
        type === 'error' ? 'linear-gradient(135deg, #ef4444, #dc2626)' :
            type === 'warning' ? 'linear-gradient(135deg, #f59e0b, #d97706)' :
                'linear-gradient(135deg, #06b6d4, #0891b2)';

    toast.style.cssText = `
        background: ${bgColor};
        color: white;
        padding: 16px 20px;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        display: flex;
        align-items: center;
        gap: 12px;
        min-width: 300px;
        animation: slideIn 0.3s ease;
        font-weight: 600;
    `;

    const icon = type === 'success' ? 'bx-check-circle' :
        type === 'error' ? 'bx-error-circle' :
            type === 'warning' ? 'bx-error' :
                'bx-info-circle';

    toast.innerHTML = `
        <i class='bx ${icon}' style="font-size: 1.5rem;"></i>
        <span style="flex: 1;">${message}</span>
    `;

    container.appendChild(toast);

    setTimeout(() => {
        toast.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

// Persistent Toast functions
function showPersistentToast(id, message, type) {
    const container = document.getElementById('toastContainer');
    if (!container) return;

    let toast = document.getElementById(id);

    if (!toast) {
        toast = document.createElement('div');
        toast.id = id;
        const bgColor = type === 'success' ? 'linear-gradient(135deg, #10b981, #059669)' :
            type === 'error' ? 'linear-gradient(135deg, #ef4444, #dc2626)' :
                type === 'warning' ? 'linear-gradient(135deg, #f59e0b, #d97706)' :
                    'linear-gradient(135deg, #06b6d4, #0891b2)';

        toast.style.cssText = `
            background: ${bgColor};
            color: white;
            padding: 16px 20px;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
            display: flex;
            align-items: center;
            gap: 12px;
            min-width: 300px;
            animation: slideIn 0.3s ease;
            font-weight: 600;
            z-index: 10001; /* Higher than normal toasts */
        `;
        container.appendChild(toast);
    }

    const icon = type === 'success' ? 'bx-check-circle' :
        type === 'error' ? 'bx-error-circle' :
            type === 'warning' ? 'bx-error' : 'bx-info-circle';

    toast.innerHTML = `
        <i class='bx ${icon}' style="font-size: 1.5rem;"></i>
        <span style="flex: 1;">${message}</span>
    `;
}

function removePersistentToast(id) {
    const toast = document.getElementById(id);
    if (toast) {
        toast.remove();
    }
}

// Fullscreen button functions
function showFullscreenButton() {
    const btn = document.getElementById('fullscreenBtn');
    if (btn) {
        btn.style.display = 'block';
    }
}

function hideFullscreenButton() {
    const btn = document.getElementById('fullscreenBtn');
    if (btn) {
        btn.style.display = 'none';
    }
}

function enterFullscreen() {
    const elem = document.documentElement;
    const requestFS = elem.requestFullscreen || elem.webkitRequestFullscreen || elem.msRequestFullscreen;
    if (requestFS) {
        requestFS.call(elem).then(() => {
            showToast('Fullscreen restored ✅', 'success');
            hideFullscreenButton();
        }).catch((err) => {
            console.error("Manual fullscreen failed:", err);
            showToast('❌ Fullscreen request denied. Please allow fullscreen!', 'error');
        });
    }
}

// Get location as soon as possible and enforce it
function requestLocation() {
    if (navigator.geolocation) {
        navigator.geolocation.getCurrentPosition(function (position) {
            userLocation.latitude = position.coords.latitude;
            userLocation.longitude = position.coords.longitude;
            console.log("Location captured:", userLocation);
        }, function (error) {
            console.warn("Location access denied or failed:", error);
            // We'll alert later in startQuiz if it's still null
        });
    } else {
        alert("Geolocation is not supported by this browser.");
    }
}
requestLocation();

//create buttons for choices
for (var i = 0; i < 4; i++) {
    var dv = document.createElement("div");
    var ch = document.createElement("button");
    ch.setAttribute("data-index", i);
    ch.setAttribute("class", "btn rounded-pill mb-2");
    ch.setAttribute("style", "background:#5f9ea0");
    choiceArray.push(ch);
    divArray.push(dv);
}

//Start Quiz function
function startQuiz() {
    showToast('Exam started! Good luck! 🎯', 'success');
    startTimer();
    buildQuestion();
    startScreenRecording();
    setupFullscreenMonitoring();

    // Show live camera preview
    const cameraPreview = document.getElementById('cameraPreview');
    if (cameraPreview) {
        cameraPreview.style.display = 'block';
        setTimeout(() => showToast('Live camera monitoring active 📹', 'info'), 1000);
    }

    // Start polling for no-person warnings
    startPersonDetectionPolling();

    // Start Background Noise Detection
    startNoiseDetection();
}

// Background Noise Detection
let noiseAudioContext;
let noiseAnalyser;
let noiseMicrophone;
let noiseScriptNode;
let noiseStream;
let noiseRecorder; // MediaRecorder instance
let noiseChunks = [];
let isNoiseRecording = false;

let noiseViolationCount = 0;
let consecutiveNoiseFrames = 0;
const NOISE_THRESHOLD = 35; // Sensitivity threshold
const NOISE_DURATION_FRAMES = 50; // Approx 2-3 seconds

async function startNoiseDetection() {
    try {
        noiseStream = await navigator.mediaDevices.getUserMedia({ audio: true });

        // Setup Analysis
        noiseAudioContext = new AudioContext();
        noiseAnalyser = noiseAudioContext.createAnalyser();
        noiseMicrophone = noiseAudioContext.createMediaStreamSource(noiseStream);
        noiseScriptNode = noiseAudioContext.createScriptProcessor(2048, 1, 1);

        noiseAnalyser.smoothingTimeConstant = 0.8;
        noiseAnalyser.fftSize = 1024;

        noiseMicrophone.connect(noiseAnalyser);
        noiseAnalyser.connect(noiseScriptNode);
        noiseScriptNode.connect(noiseAudioContext.destination);

        // Setup Recording (but don't start yet)
        noiseRecorder = new MediaRecorder(noiseStream);

        noiseRecorder.ondataavailable = function (e) {
            if (e.data.size > 0) noiseChunks.push(e.data);
        };

        noiseRecorder.onstop = function () {
            // Upload evidence if we have chunks
            if (noiseChunks.length > 0) {
                const blob = new Blob(noiseChunks, { type: 'audio/webm' });
                uploadNoiseEvidence(blob);
                noiseChunks = [];
            }
        };

        noiseScriptNode.onaudioprocess = function () {
            if (secondsLeft <= 0) {
                stopNoiseDetection();
                return;
            }

            var array = new Uint8Array(noiseAnalyser.frequencyBinCount);
            noiseAnalyser.getByteFrequencyData(array);

            // Calculate average volume
            let values = 0;
            for (let i = 0; i < array.length; i++) values += array[i];
            let average = values / array.length;

            if (average > NOISE_THRESHOLD) {
                consecutiveNoiseFrames++;

                // If noise starts, start recording evidence
                if (!isNoiseRecording && consecutiveNoiseFrames > 10) {
                    startEvidenceRecording();
                }

                if (consecutiveNoiseFrames > NOISE_DURATION_FRAMES) {
                    handleNoiseViolation();
                    consecutiveNoiseFrames = 0;
                }
            } else {
                consecutiveNoiseFrames = Math.max(0, consecutiveNoiseFrames - 1);

                // If noise stops for a while, stop recording
                if (isNoiseRecording && consecutiveNoiseFrames === 0) {
                    stopEvidenceRecording();
                }
            }
        }
        console.log("Background noise detection started");

    } catch (err) {
        console.error("Noise detection failed:", err);
        showToast('⚠️ Audio monitoring failed.', 'warning');
    }
}

function startEvidenceRecording() {
    if (noiseRecorder && noiseRecorder.state === 'inactive') {
        isNoiseRecording = true;
        noiseChunks = [];
        noiseRecorder.start();
        console.log("Started recording noise evidence...");
    }
}

function stopEvidenceRecording() {
    if (noiseRecorder && noiseRecorder.state === 'recording') {
        isNoiseRecording = false;
        noiseRecorder.stop();
        console.log("Stopped recording noise evidence.");
    }
}

function uploadNoiseEvidence(blob) {
    const now = Date.now();
    if (now - lastNoiseViolationTime > 15000) {
        console.log("Discarding audio clip (no violation triggered)");
        return;
    }

    const formData = new FormData();
    formData.append('audio_evidence', blob, 'noise_evidence.webm');
    formData.append('sessionId', examSessionId);

    $.ajax({
        type: 'POST',
        url: '/upload_audio_violation',
        data: formData,
        processData: false,
        contentType: false,
        success: function (response) {
            console.log("Noise evidence uploaded:", response);
        },
        error: function (err) {
            console.error("Failed to upload noise evidence:", err);
        }
    });
}

function stopNoiseDetection() {
    if (noiseScriptNode) noiseScriptNode.disconnect();
    if (noiseAnalyser) noiseAnalyser.disconnect();
    if (noiseMicrophone) noiseMicrophone.disconnect();
    if (noiseStream) noiseStream.getTracks().forEach(track => track.stop());
}

let lastNoiseViolationTime = 0;

function handleNoiseViolation() {
    const now = Date.now();
    if (now - lastNoiseViolationTime < 10000) return;

    lastNoiseViolationTime = now;
    noiseViolationCount++;

    console.warn("High background noise detected!");
    showToast('⚠️ High background noise detected! maintain silence.', 'warning');

    // Force current recording fragment to upload
    if (isNoiseRecording) {
        stopEvidenceRecording();
        setTimeout(() => {
            if (secondsLeft > 0) startEvidenceRecording();
        }, 1000);
    } else {
        // If not already recording, start and stop after a few seconds to get a clip
        startEvidenceRecording();
        setTimeout(stopEvidenceRecording, 5000);
    }

    reportViolation("Background Noise", "Sustained high volume detected via microphone");
}

// Poll for camera absence
let personPollingInterval = null;
let absenceOverlay, absenceTimer, absenceProgressBar, absenceIconContainer, absenceTitle, absenceMessage;
let lastAudioPlayTime = 0;

function playWarningAudio() {
    try {
        const now = Date.now();
        if (now - lastAudioPlayTime < 1000) return; // Limit to once per second
        lastAudioPlayTime = now;

        const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        const oscillator = audioCtx.createOscillator();
        const gainNode = audioCtx.createGain();

        oscillator.connect(gainNode);
        gainNode.connect(audioCtx.destination);

        oscillator.type = 'sine';
        oscillator.frequency.setValueAtTime(440, audioCtx.currentTime); // A4 note
        gainNode.gain.setValueAtTime(0.1, audioCtx.currentTime);
        gainNode.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.5);

        oscillator.start();
        oscillator.stop(audioCtx.currentTime + 0.5);
    } catch (e) {
        console.warn("Audio play failed:", e);
    }
}

// Initialize elements when DOM is ready
function initializeAbsenceElements() {
    absenceOverlay = document.getElementById('cameraAbsenceOverlay');
    absenceTimer = document.getElementById('absenceTimer');
    absenceProgressBar = document.getElementById('absenceProgressBar');
    absenceIconContainer = document.getElementById('absenceIconContainer');
    absenceTitle = document.getElementById('absenceTitle');
    absenceMessage = document.getElementById('absenceMessage');
}

// Initialize elements immediately (script loads at end of body, so DOM should be ready)
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeAbsenceElements);
} else {
    initializeAbsenceElements();
}

function startPersonDetectionPolling() {
    if (personPollingInterval) {
        console.log("Person detection polling already started");
        return;
    }

    // Ensure elements are initialized
    if (!absenceOverlay) {
        initializeAbsenceElements();
    }

    console.log("Starting person detection polling...", {
        overlay: !!absenceOverlay,
        timer: !!absenceTimer,
        progressBar: !!absenceProgressBar
    });

    personPollingInterval = setInterval(() => {
        $.ajax({
            type: "GET",
            url: "/api/check_person_status",
            success: function (response) {
                console.log("Person status check:", response.status, response);
                // 1. Alert Phase (Toast Only)
                if (response.status === 'alert_phase') {
                    if (absenceOverlay) absenceOverlay.style.display = 'none';
                    showPersistentToast('person-warning', response.message, 'warning');
                }
                // 2. Countdown Phase (Overlay)
                else if (response.status === 'countdown_phase') {
                    // Play warning sound
                    playWarningAudio();

                    // Hide toast, show overlay
                    removePersistentToast('person-warning');
                    if (absenceOverlay) {
                        absenceOverlay.style.display = 'flex';

                        // Update Timer
                        if (absenceTimer) {
                            absenceTimer.textContent = `00:${response.remaining.toString().padStart(2, '0')}`;
                        }

                        // Update Progress Bar
                        if (absenceProgressBar) {
                            const percent = (response.remaining / response.total_countdown) * 100;
                            absenceProgressBar.style.width = `${percent}%`;
                        }

                        // Critical Warning (< 15s)
                        if (response.remaining <= 15) {
                            // Update Visuals to Red/Critical
                            if (absenceIconContainer) {
                                absenceIconContainer.style.background = 'linear-gradient(135deg, #ef4444, #dc2626)';
                                absenceIconContainer.innerHTML = "<i class='bx bxs-error'></i>";
                                absenceIconContainer.style.boxShadow = '0 0 30px rgba(239, 68, 68, 0.4)';
                            }
                            if (absenceProgressBar) absenceProgressBar.style.backgroundColor = '#ef4444';
                            if (absenceTitle) absenceTitle.innerHTML = "WARNING: Exam will be terminated";
                            if (absenceTitle) absenceTitle.style.color = '#ef4444';
                            if (absenceTimer) absenceTimer.style.color = '#ef4444';
                        } else {
                            // Standard Warning (Yellow/Orange)
                            if (absenceIconContainer) {
                                absenceIconContainer.style.background = 'linear-gradient(135deg, #f59e0b, #d97706)';
                                absenceIconContainer.innerHTML = "<i class='bx bxs-zap'></i>";
                                absenceIconContainer.style.boxShadow = '0 0 30px rgba(245, 158, 11, 0.4)';
                            }
                            if (absenceProgressBar) absenceProgressBar.style.backgroundColor = '#f59e0b';
                            if (absenceTitle) absenceTitle.innerHTML = "Return to camera view within:";
                            if (absenceTitle) absenceTitle.style.color = 'white';
                            if (absenceTimer) absenceTimer.style.color = 'white';
                        }
                    }
                }
                // 3. Terminated
                else if (response.status === 'terminated') {
                    window.location.href = "/exam_terminated";
                }
                // 4. OK / Grace
                else {
                    if (absenceOverlay) absenceOverlay.style.display = 'none';
                    removePersistentToast('person-warning');
                }
            },
            error: function (xhr, status, error) {
                console.error("Failed to check person status:", error);
            }
        });
    }, 1000);
}

function checkUserPresence() {
    console.log("User clicked 'I'm Back'. Resetting absence timer...");

    // Immediate UI feedback
    if (absenceOverlay) absenceOverlay.style.display = 'none';

    $.ajax({
        type: "POST",
        url: "/api/reset_absence",
        success: function (response) {
            console.log("Absence timer reset successfully.");
            // Further validation will happen in the next polling interval
        },
        error: function (xhr, status, error) {
            console.error("Failed to reset absence timer via API:", error);
        }
    });
}

function stopPersonDetectionPolling() {
    if (personPollingInterval) {
        clearInterval(personPollingInterval);
        personPollingInterval = null;
    }
}

// "I'm Back" Button Handler
window.checkUserPresence = function () {
    const btn = document.getElementById('imBackButton');
    const originalText = btn.textContent;
    btn.textContent = "Checking...";
    btn.disabled = true;

    // Force a poll/wait for poll
    setTimeout(() => {
        // We rely on the next poll to clear it if successful
        // But we can give visual feedback
        $.ajax({
            type: "GET",
            url: "/api/check_person_status",
            success: function (response) {
                if (response.status === 'ok' || response.status === 'grace_period') {
                    // Success - overlay will hide automatically by polling
                    showToast('Welcome back! 👋', 'success');
                } else {
                    // Failed
                    btn.textContent = "Still No Face Detected";
                    btn.style.background = "#ef4444";
                    setTimeout(() => {
                        btn.textContent = originalText;
                        btn.style.background = "linear-gradient(135deg, #06b6d4, #3b82f6)";
                        btn.disabled = false;
                    }, 1500);
                }
            }
        });
    }, 500);
};

function setupFullscreenMonitoring() {
    // Monitor for fullscreen exit
    document.onfullscreenchange = document.onwebkitfullscreenchange = document.onmsfullscreenchange = function () {
        const isFS = document.fullscreenElement || document.webkitFullscreenElement || document.msFullscreenElement;

        // Only trigger if quiz is active and not finished
        if (!isFS && secondsLeft > 0 && questionIndex <= totalQuestions - 1) {
            console.warn("Fullscreen violation detected!");
            showToast('⚠️ Fullscreen Exit Detected! Violation logged.', 'error');
            // Report violation to backend - the backend will decide if it's a warning or termination
            reportViolation("Fullscreen Exit", "User manually exited fullscreen mode");
        }
    };
}

function reportViolation(type, details) {
    $.ajax({
        type: "POST",
        url: "/api/violation",
        contentType: "application/json",
        data: JSON.stringify({
            type: type,
            details: details,
            sessionId: examSessionId
        }),
        success: function (response) {
            console.log("Violation response:", response.action);
            if (response.action === "warning") {
                // First time warning - show toast and re-request fullscreen
                showToast('⚠️ FINAL WARNING: ' + response.message, 'warning');

                // Wait a moment for toast to be visible, then re-request fullscreen
                setTimeout(() => {
                    const elem = document.documentElement;
                    const requestFS = elem.requestFullscreen || elem.webkitRequestFullscreen || elem.msRequestFullscreen;
                    if (requestFS) {
                        requestFS.call(elem).then(() => {
                            showToast('Fullscreen restored ✅', 'success');
                            hideFullscreenButton();
                        }).catch((err) => {
                            console.error("Auto-restoration of fullscreen failed:", err);
                            showToast('⚠️ Click the button to continue in fullscreen!', 'warning');
                            showFullscreenButton();
                        });
                    }
                }, 500);
            } else if (response.action === "terminated") {
                // Second time (or other critical) - terminate immediately
                showToast('❌ Exam terminated due to violations!', 'error');
                setTimeout(() => window.location.href = "/exam_terminated", 1500);
            }
        },
        error: function (xhr, status, error) {
            console.error("Failed to log violation:", error);
        }
    });
}

async function startScreenRecording() {
    try {
        const displayStream = await navigator.mediaDevices.getDisplayMedia({
            video: { mediaSource: "screen" }
        });

        // Request microphone for background audio
        let combinedStream = displayStream;
        try {
            const audioStream = await navigator.mediaDevices.getUserMedia({ audio: true });
            combinedStream = new MediaStream([
                ...displayStream.getVideoTracks(),
                ...audioStream.getAudioTracks()
            ]);
            console.log("Background audio integrated with screen recording");
        } catch (audioErr) {
            console.warn("Could not access microphone for screen recording audio:", audioErr);
        }

        mediaRecorder = new MediaRecorder(combinedStream);
        recordedChunks = [];

        mediaRecorder.ondataavailable = function (e) {
            if (e.data.size > 0) {
                recordedChunks.push(e.data);
            }
        };

        mediaRecorder.onstop = function () {
            const blob = new Blob(recordedChunks, { type: 'video/webm' });
            uploadRecording(blob);
            stream.getTracks().forEach(track => track.stop());
        };

        mediaRecorder.start();
        console.log("Screen recording started");

        // TRIGGER FULLSCREEN AFTER SCREEN SHARE SUCCESS
        const elem = document.documentElement;
        if (elem.requestFullscreen) {
            elem.requestFullscreen().then(() => {
                showToast('Fullscreen mode activated', 'success');
            }).catch(err => {
                console.log("Full screen denied:", err);
                showToast('Fullscreen denied - please enable it', 'warning');
            });
        } else if (elem.webkitRequestFullscreen) {
            elem.webkitRequestFullscreen();
        } else if (elem.msRequestFullscreen) {
            elem.msRequestFullscreen();
        }

        // Start polling for exam status (to handle backend termination)
        startStatusPolling();
    } catch (err) {
        console.error("Error starting screen recording:", err);
        alert("Screen recording is required for this exam. Please allow screen sharing.");
    }
}

function startStatusPolling() {
    const statusInterval = setInterval(function () {
        if (secondsLeft <= 0 || questionIndex > totalQuestions - 1) {
            clearInterval(statusInterval);
            return;
        }

        $.ajax({
            type: "GET",
            url: "/check_exam_status",
            success: function (response) {
                if (response.status === 'terminated') {
                    clearInterval(statusInterval);
                    window.location.href = "/exam_terminated";
                }
            },
            error: function (xhr, status, error) {
                console.error("Status check failed:", error);
            }
        });
    }, 3000); // Check every 3 seconds
}

function uploadRecording(blob) {
    const formData = new FormData();
    formData.append('recording', blob, 'recording.webm');
    formData.append('sessionId', examSessionId);

    $.ajax({
        type: 'POST',
        url: '/upload_recording',
        data: formData,
        processData: false,
        contentType: false,
        success: function (response) {
            console.log("Recording uploaded successfully:", response);
        },
        error: function (err) {
            console.error("Recording upload failed:", err);
        }
    });
}

//function to start timer when quiz starts
function startTimer() {

    var timeInterval = setInterval(function () {

        secondsLeft--;

        timer.textContent = "Time : " + secondsLeft + " sec";
        //        if(secondsLeft <= 60){
        //            timer.textContent = "Time : 1 min";
        //        }

        if (secondsLeft <= 0 || (questionIndex > totalQuestions - 1)) {

            resultDiv.style.display = "none";
            quizContent.style.display = "none";
            viewResult();
            clearInterval(timeInterval);
            timer.textContent = "";
        }

    }, 1000);
}


function buildQuestion() {

    //hides start page content
    questionEl.style.display = "none";
    mainContent.style.display = "none";
    quizContent.style.display = "none";

    if (questionIndex > totalQuestions - 1) {
        return;
    }
    else {
        ans = questions[questionIndex].answer;

        //Display Question
        questionEl.innerHTML = questions[questionIndex].title;
        questionEl.setAttribute("class", "text-left");
        questionEl.style.display = "block";

        for (var j = 0; j < 4; j++) {
            var index = choiceArray[j].getAttribute("data-index");
            choiceArray[j].textContent = (+index + 1) + ". " + questions[questionIndex].choices[index];
            divArray[j].appendChild(choiceArray[j]);
            quizContent.appendChild(divArray[j]);
        }

    }
    quizContent.style.display = "block"; // Display options
}

// Event Listener for options buttons
quizContent.addEventListener("click", function (event) {

    var element = event.target;
    var userAnswer = element.textContent;
    var userOption = userAnswer.substring(3, userAnswer.length);

    if (userOption === ans) {
        correct++;

        resultDiv.style.display = "block";
    }
    else {
        secondsLeft -= 10;

        setTimeout(function () {
            resultDiv.textContent = "";
        }, 500);
    }

    questionIndex++;
    buildQuestion();
});


//Function to show score when quiz completes
function viewResult() {

    questionEl.innerHTML = "Great Job! Your Test Completed!";
    questionEl.style.display = "block";


    var scoreButton = document.createElement("button");     //Submit User score
    scoreButton.setAttribute("class", "btn rounded-pill mb-2 ml-3 mt-2");
    scoreButton.setAttribute("style", "background:#5f9ea0");
    scoreButton.textContent = "Submit";
    completeTest.appendChild(scoreButton);

    scoreButton.addEventListener("click", function () {
        // Stop recording if it's running
        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            mediaRecorder.stop();
        }

        var inputData = correct;
        $.ajax({
            type: "POST",
            url: "/exam",
            contentType: "application/json",
            data: JSON.stringify({
                input: inputData,
                location: userLocation,
                sessionId: examSessionId
            }),
            success: function (response) {
                console.log("Submission response:", response);

                if (response['link'] === 'main') {
                    alert("Session expired. Please log in again.");
                    window.location.href = "/main";
                } else {
                    // Redirect to results page with encoded parameters
                    window.location.href = "/" + encodeURIComponent(response['link']) + "/" + encodeURIComponent(response['output']);
                }
            },
            error: function (xhr, status, error) {
                console.error("Submission failed:", error);
                alert("Submission failed. Please check your connection and try again.");
            }
        });
    });
}
/*
//Function to store highscores
function storeScores(event){

    event.preventDefault();
    var userName = document.querySelector("#nameInput").value.trim();

    if(userName === null || userName === '') {
        alert("Please enter user name");
        return;
     }

      //Create user object for storing highscore
        var user = {
            name : userName,
            score : correct
        }

        console.log(user);

        previousScores = JSON.parse(localStorage.getItem("previousScores"));    //get User highscores array in localStorage if exists

        if(previousScores){
            previousScores.push(user); //Push new user scores in array in localStorage
        }
        else{
            previousScores = [user];    //If No user scores stored in localStorage, create array to store user object
        }

        // set new submission
        localStorage.setItem("previousScores",JSON.stringify(previousScores));

        showHighScores(); // Called function to display highscores

}
*/
//Start button event listener on start page which starts quiz
$(document).ready(function () {
    startButton.addEventListener("click", function () {
        $.ajax({
            type: "POST",
            url: "/exam",
            contentType: "application/json",
            data: JSON.stringify({ input: '' }),
            success: function (response) {
                startQuiz();
            },
            error: function (xhr, status, error) {
                console.error("Failed to start exam session:", error);
                startQuiz();
            }
        });
    });
});
