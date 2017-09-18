while true; do
	python app.py &
	pid=$!
	sleep 10
	kill $pid
        sleep 1
done
