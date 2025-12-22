# How to Run the Online Exam Portal

## Step 1: Install Dependencies
```bash
cd "The-Online-Exam-Proctor"
pip install -r requirements.txt
```

## Step 2: Set Up Environment Variables
Create a `.env` file in the project root with:
```
MONGO_URI=your_mongodb_connection_string
SECRET_KEY=your_secret_key_here
MAIL_SERVER=smtp.gmail.com
MAIL_PORT=587
MAIL_USE_TLS=True
MAIL_USERNAME=your_email@gmail.com
MAIL_PASSWORD=your_app_password
GOOGLE_CLIENT_ID=your_google_client_id
GOOGLE_CLIENT_SECRET=your_google_client_secret
```

## Step 3: Run the Application
```bash
python app.py
```

The application will start at: `http://127.0.0.1:5000`

## Step 4: Test the Countdown Feature

1. **Login** as a student (default: student@test.com / password)
2. **Start the exam** by clicking "Start Quiz"
3. **Move away from the camera** - the countdown will start:
   - **0-10 seconds**: Grace period (no warning)
   - **10-30 seconds**: Warning toast appears
   - **30-60 seconds**: Full-screen countdown overlay appears
   - **After 60 seconds**: Exam terminates automatically

## Troubleshooting

### If the countdown overlay doesn't appear:
1. Check browser console for JavaScript errors (F12)
2. Ensure camera permissions are granted
3. Verify the camera is working and detecting faces
4. Check that `/api/check_person_status` endpoint is responding

### If the application won't start:
1. Check Python version (requires Python 3.7+)
2. Verify all dependencies are installed
3. Check MongoDB connection string
4. Ensure port 5000 is not in use

