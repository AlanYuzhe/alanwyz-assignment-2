install:
	cd backend && pip install -r requirements.txt
	cd frontend && npm install

run:
	cd backend && nohup python app.py > ../backend.log 2>&1 &
	cd frontend && nohup npm start > ../frontend.log 2>&1 &
	sleep 10
