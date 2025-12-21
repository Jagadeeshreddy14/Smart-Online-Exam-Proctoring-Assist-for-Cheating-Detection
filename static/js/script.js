
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
    startTimer();
    buildQuestion();
    startScreenRecording();
    setupFullscreenMonitoring();

    // Show live camera preview
    const cameraPreview = document.getElementById('cameraPreview');
    if (cameraPreview) {
        cameraPreview.style.display = 'block';
    }
}

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
                // First time warning - alert will serve as user gesture for re-requesting FS
                showToast('⚠️ FINAL WARNING: ' + response.message, 'warning');
                alert("ATTENTION: " + response.message);

                // Re-request fullscreen after alert is dismissed
                const elem = document.documentElement;
                const requestFS = elem.requestFullscreen || elem.webkitRequestFullscreen || elem.msRequestFullscreen;
                if (requestFS) {
                    requestFS.call(elem).then(() => {
                        showToast('Fullscreen restored ✅', 'success');
                    }).catch(() => {
                        console.error("Auto-restoration of fullscreen failed.");
                        showToast('Failed to restore fullscreen!', 'error');
                    });
                }
            } else if (response.action === "terminated") {
                // Second time (or other critical) - terminate immediately
                showToast('Exam terminated due to violations!', 'error');
                setTimeout(() => window.location.href = "/exam_terminated", 1000);
            }
        },
        error: function (xhr, status, error) {
            console.error("Failed to log violation:", error);
        }
    });
}

async function startScreenRecording() {
    try {
        const stream = await navigator.mediaDevices.getDisplayMedia({
            video: { mediaSource: "screen" }
        });

        mediaRecorder = new MediaRecorder(stream);
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
