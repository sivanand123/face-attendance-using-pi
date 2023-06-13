############################################# IMPORTING ################################################
import tkinter as tk
from tkinter import ttk
import cv2,os
import csv
import tflite_runtime.interpreter as tflite
import numpy as np
import datetime
import time
import serial
import pynmea2
import sqlite3
import imutils
from imutils.video import VideoStream

############################################# FUNCTIONS ################################################

ser = serial.Serial('/dev/ttyAMA0', baudrate=9600, timeout=1)

while True:
    try:
        
        line = ser.readline().decode('ascii', errors='replace').strip()
        
        
        if line.startswith('$'):
            
            msg = pynmea2.parse(line)
            if msg.sentence_type == 'GGA':
                lat = msg.latitude
                lat_dir = msg.lat_dir
                lon = msg.longitude
                lon_dir = msg.lon_dir
                alt = msg.altitude
                
                
                print('Latitude: %s%s' % (lat, lat_dir))
                print('Longitude: %s%s' % (lon, lon_dir))
                print('Altitude: %s meters' % alt)
                break
                
    except (serial.SerialException, pynmea2.ParseError):
        
        print('Error reading from serial port')

locations = [(17.0664, 81.8733), (17.06234, 81.8733), (17.0651, 81.8715), (17.0652, 81.8754)]


my_location = (17.0664, 81.8733)


if my_location[0] < min(coord[0] for coord in locations) or my_location[0] > max(coord[0] for coord in locations) or my_location[1] < min(coord[1] for coord in locations) or my_location[1] > max(coord[1] for coord in locations):
    print("Location out of range.")
else:
    
    def tick():
        time_string = time.strftime('%H:%M:%S')
        clock.config(text=time_string)
        clock.after(200,tick)

    def check_haarcascadefile():
        exists = os.path.isfile("haarcascade_frontalface_default.xml")
        if exists:
            pass
        else:
            mess._show(title='Some file missing', message='Please contact us for help')
            window.destroy()


    def TrackImages():
        harcascadePath = "haarcascade_frontalface_default.xml"
        faceCascade = cv2.CascadeClassifier(harcascadePath)
        detect2 = cv2.CascadeClassifier('haarcascade_eye.xml')
        classes = ["1001 Aditya","1002 Bharath","1003 ChaitanayaKiran","1004 Chandu","1005 Dinesh","1006 Farooq","1007 Ganesh","1008 Haswanth","1009 Karthik","1010 Kumarkalyan","1011 Phanindra","1012 Praneeth","1013 Prasad","1014 PushpaRaj","1016 Ravindra","1017 SaiMahesh","1018 Satya_sai","1019 Sivanand","1020 SureshReddy","1021 Vijay","1022 William"]
        attendance = []
        all_attendance = []
        last_records=[]
        # Load Anti-Spoofing Model
        interpreter_spoof = tflite.Interpreter(model_path="finalspoof_model.tflite")
        interpreter_spoof.allocate_tensors()
        input_details = interpreter_spoof.get_input_details()
        output_details = interpreter_spoof.get_output_details()

        #Load Face-recognition model
        interpreter_FR = tflite.Interpreter(model_path="face65rim.tflite")
        interpreter_FR.allocate_tensors()
        input_details_FR= interpreter_FR.get_input_details()
        output_details_FR = interpreter_FR.get_output_details()
        conn = sqlite3.connect('faces.db')
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS attendanceface
                    (id TEXT, name TEXT, data TEXT, time TEXT, location TEXT)''')
        #cam = cv2.VideoCapture(0)
        vs = VideoStream(usePiCamera=True).start()
        time.sleep(2.0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        print("-----This is initial attendance printing----------")
        
        while True:
            frame = vs.read()
            frame = imutils.resize(frame, width=500)
            #ret, frame = cam.read()
            faces = faceCascade.detectMultiScale(frame, 1.2, 5)
            for (x, y, w, h) in faces:
                face = frame[y - 5:y + h + 5, x - 5:x + w + 5]
                resized_face = cv2.resize(face, (160, 160))
                normalized_face = resized_face.astype("float32") / 255.0
                input_data_spoof = np.expand_dims(normalized_face, axis=0)
                interpreter_spoof.set_tensor(input_details[0]['index'], input_data_spoof)
                interpreter_spoof.invoke()
                output_data_spoof = interpreter_spoof.get_tensor(output_details[0]['index'])
                if output_data_spoof[0][0] > 0.3:
                    label = "spoof" 
                    confidence = output_data_spoof[0][0]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255) , 2)
                    cv2.putText(frame, f"{label} ({confidence:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 255), 2)
                else :
                    eyes = detect2.detectMultiScale(face, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
                    if len(eyes) > 0:
                        face_img = cv2.resize(face, (224, 224))
                        img = np.expand_dims(face_img.astype('float32')/255.0, axis=0)
                        
                        
                        interpreter_FR.set_tensor(input_details_FR[0]['index'], img)
                        interpreter_FR.invoke()
                        
                        
                        output = interpreter_FR.get_tensor(output_details_FR[0]['index'])
                        max_prob = np.max(output)
                        if max_prob >= 0.9:
                            prediction = np.argmax(output)   
                            class_name = classes[prediction]
                            ID,name = class_name.split()
                            ts = time.time()
                            date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
                            timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                            # g = geocoder.ip('me')
                            # lat,long = g.latlng
                            # location = str(lat) +','+ str(long)
                            location = str(lat) +',' +str(lon)
                            print(location)
                            attendance = [str(ID), '', name, '', str(date), '', str(timeStamp),'',str(location)]
                            all_attendance.append(attendance)
                            print(attendance)
                            id_last_record = {}
                            id_counts={}
                            for attendance in all_attendance:
                                id = attendance[0]
                                if id in id_counts:
                                    id_counts[id] += 1
                                else:
                                    id_counts[id] = 1
                            for attendance in reversed(all_attendance):
                                id = attendance[0]
                                if id_counts[id] > 7:
                                    if id not in id_last_record:
                                        id_last_record[id] = attendance
                                        id_counts[id] = -1
                            last_records = list(id_last_record.values())
                        else:
                            Id = 'Unknown'
                            name = str(Id)
                        confidence = output_data_spoof[0][0]
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(frame, f"{name} ({confidence:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255,0), 2)
            cv2.imshow('Taking Attendance', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        if last_records:
            print('......................')
            print('.....................')
            print(last_records)
            print('.......................')
            print('......................')
        if last_records:
            ts = time.time()
            date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
            file_path = "Attendance/Attendance_" + date + ".csv"
            exists = os.path.isfile(file_path)
            
            with open(file_path, 'a+', newline='') as csvFile1:
                writer = csv.writer(csvFile1)
                if not exists: 
                    #writer.writerow(["ID", "NAME", "DATE", "TIME", "LOCATION"])
                    first_record = last_records[0]
                    id = first_record[0]
                    name = first_record[2]
                    date_time = datetime.datetime.strptime(first_record[4] + " " + first_record[6], '%d-%m-%Y %H:%M:%S')
                    date = date_time.strftime('%d-%m-%Y')
                    current_time = date_time.strftime('%H:%M:%S')
                    location = first_record[8]
                    writer.writerow([id, name, date, current_time, location])
                for record in last_records:
                    id = record[0]
                    name = record[2]
                    date_time = datetime.datetime.strptime(record[4] + " " + record[6], '%d-%m-%Y %H:%M:%S')
                    date = date_time.strftime('%d-%m-%Y')
                    current_time = date_time.strftime('%H:%M:%S')
                    location = record[8]
                    writer.writerow([id, name, date, current_time, location])
            csvFile1.close()

            
            with open("Attendance/Attendance_" + date + ".csv", 'r') as csvFile1:
                reader1 = csv.reader(csvFile1)
                for i, lines in enumerate(reader1, start=1):
                    
                    iidd = str(lines[0])
                    values = (str(lines[1]), str(lines[2]), str(lines[3]), str(lines[4]))
                    print(iidd, values)
                    c.execute('INSERT INTO attendanceface VALUES (?,?,?,?,?)', (iidd,) + values)
                    c.execute('SELECT COUNT(*) FROM attendanceface WHERE id = ?', (iidd,))
                    tv.insert('', 0, text=iidd, values=values)
                conn.commit()
                conn.close()
                csvFile1.close()

        
        cv2.destroyAllWindows()
        vs.stop()

        

    ######################################## USED STUFFS ############################################
        
    global key
    key = ''

    ts = time.time()
    date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
    day,month,year=date.split("-")

    mont={'01':'January',
          '02':'February',
          '03':'March',
          '04':'April',
          '05':'May',
          '06':'June',
          '07':'July',
          '08':'August',
          '09':'September',
          '10':'October',
          '11':'November',
          '12':'December'
          }

    ######################################## GUI FRONT-END ###########################################

    window = tk.Tk()
    window.geometry("1280x720")
    window.resizable(True,False)
    window.title("Attendance System")
    window.configure(background='#262523')

    frame1 = tk.Frame(window, bg="#00aeff")
    frame1.place(relx=0.3, rely=0.17, relwidth=0.5, relheight=0.80)


    message3 = tk.Label(window, text="Face Recognition Based Attendance System" ,fg="white",bg="#262523" ,width=55 ,height=1,font=('times', 29, ' bold '))
    message3.place(x=10, y=10)

    frame3 = tk.Frame(window, bg="#c4c6ce")
    frame3.place(relx=0.52, rely=0.09, relwidth=0.09, relheight=0.07)

    frame4 = tk.Frame(window, bg="#c4c6ce")
    frame4.place(relx=0.36, rely=0.09, relwidth=0.16, relheight=0.07)

    datef = tk.Label(frame4, text = day+"-"+mont[month]+"-"+year+"  |  ", fg="orange",bg="#262523" ,width=55 ,height=1,font=('times', 22, ' bold '))
    datef.pack(fill='both',expand=1)

    clock = tk.Label(frame3,fg="orange",bg="#262523" ,width=55 ,height=1,font=('times', 22, ' bold '))
    clock.pack(fill='both',expand=1)
    tick()



    ################## TREEVIEW ATTENDANCE TABLE ####################

    tv= ttk.Treeview(frame1,height =13,columns = ('name','date','time','location'))
    tv.column('#0',width=82)
    tv.column('name',width=130)
    tv.column('date',width=133)
    tv.column('time',width=133)
    tv.column('location',width=133)
    tv.grid(row=2,column=0,padx=(0,0),pady=(150,0),columnspan=5)
    tv.heading('#0',text ='ID')
    tv.heading('name',text ='NAME')
    tv.heading('date',text ='DATE')
    tv.heading('time',text ='TIME')
    tv.heading('location',text='LOCATION')

    ###################### SCROLLBAR ################################

    scroll=ttk.Scrollbar(frame1,orient='vertical',command=tv.yview)
    scroll.grid(row=2,column=5,padx=(0,100),pady=(150,0),sticky='ns')
    tv.configure(yscrollcommand=scroll.set)

    ###################### BUTTONS ##################################

    trackImg = tk.Button(frame1, text="Take Attendance", command=TrackImages  ,fg="black"  ,bg="yellow"  ,width=35  ,height=1,
                         activebackground = "white" ,font=('times', 15, ' bold '))
    trackImg.place(x=100,y=50)
    quitWindow = tk.Button(frame1, text="Quit", command=window.destroy  ,fg="black"  ,bg="red"  ,width=35 ,height=1,
                           activebackground = "white" ,font=('times', 15, ' bold '))
    quitWindow.place(x=115, y=450)

    ##################### END ######################################

    window.mainloop()


