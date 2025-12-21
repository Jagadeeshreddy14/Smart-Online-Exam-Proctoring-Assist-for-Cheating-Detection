
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

// Get location as soon as possible
if (navigator.geolocation) {
    navigator.geolocation.getCurrentPosition(function (position) {
        userLocation.latitude = position.coords.latitude;
        userLocation.longitude = position.coords.longitude;
        console.log("Location captured:", userLocation);
    }, function (error) {
        console.warn("Location access denied or failed:", error);
    });
}

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

    var elem = document.documentElement;
    if (elem.requestFullscreen) {
        elem.requestFullscreen();
    } else if (elem.webkitRequestFullscreen) { /* Safari */
        elem.webkitRequestFullscreen();
    } else if (elem.msRequestFullscreen) { /* IE11 */
        elem.msRequestFullscreen();
    }

    startTimer();
    buildQuestion();
    startScreenRecording();
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
    } catch (err) {
        console.error("Error starting screen recording:", err);
        alert("Screen recording is required for this exam. Please allow screen sharing.");
    }
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

        // Trigger Full Screen immediately on user click
        var elem = document.documentElement;
        if (elem.requestFullscreen) {
            elem.requestFullscreen().catch(err => console.log("Full screen denied:", err));
        } else if (elem.webkitRequestFullscreen) { /* Safari */
            elem.webkitRequestFullscreen();
        } else if (elem.msRequestFullscreen) { /* IE11 */
            elem.msRequestFullscreen();
        }

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
                // Even if the initial call fails, let's try starting the quiz
                startQuiz();
            }
        });
    });
});
