# -*- coding: utf-8 -*-
"""
Created on Tue Nov 03 16:46:15 2015

@author: JMS
"""

import base64
import datetime
import re
import socket
import time
import threading
import slugid as slugid
import binascii

from SetupXMLFileManager import *
from bitarray import *
from PyQt5.QtCore import QObject, pyqtSignal



class MicroAdapterAtlas(QObject):
    """
    Adapter class for SEM microscope
    
    """
    WS_MAGIC_STRING = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
    iplocal = "127.0.0.1"
    ipAtlas = ""
    port = 8098
    status = ''
    current_log_file = ''
    connected = False
    secure_mode = True
    scale = "micrometer"
    new_message = pyqtSignal(str)
    warning = pyqtSignal(str)
    request = pyqtSignal(str)
    critical = pyqtSignal(str)
    logger = []
    ping_lock = threading.Lock()

    def __init__(self, parent = None):
        QObject.__init__(self)
        self.currentIP = self.iplocal
        self.address = "http://"+str(self.currentIP)+":"+str(self.port)+"/"  

    def setLogger(self,ilogger):
        self.logger = ilogger

    def changeIp(self,o_iplm):
        self.currentIP = o_iplm 
        self.address = "http://"+str(self.currentIP)+":"+str(self.port)+"/"  
                      
    def changeIpLocal(self):
        self.currentIP = self.iplocal         
        self.address = "http://"+str(self.currentIP)+":"+str(self.port)+"/"
    
    def getIpLocal(self):
        return self.iplocal
    
    def getIpAtlas(self):
        return self.ipAtlas

    def changeIpAtlas(self):
        self.currentIP = self.ipAtlas
        self.address = "http://"+self.currentIP+":"+str(self.port)+"/"
        
    def changePort(self,o_port):
        self.port = o_port
        self.address = "http://"+self.currentIP+":"+str(self.port)+"/"

    def getNetworkIp(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        s.connect(('<broadcast>', 0))
        return s.getsockname()[0]
        
    def is_connected(self):
        return self.connected

    def connect(self):
        """
        Connects to the MSITE4A server
        :return:
        """
        # establish communication to microscope
        ####################################<<<<<<<<<<<<<<<<
        ## prepare package
        #  FIN 1
        #  RSV1,RSV2,RSV3 000
        #  Opcode 0xB
        a = bitarray('10001011')
        #  Payload 9 bytes of 0  
        a.extend(9*8*'0')
        error = ''
        self.logger.info("Sending information request to msite4A:\n")

        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            client_socket.connect((self.currentIP, int(self.port))) 
            client_socket.sendall(a.tobytes());
            data,header = self.recv_timeout(client_socket,30); # wait 30 seconds for the initial connection
        except socket.timeout as msg:
            error = str(msg)
            error = error + "\n Atlas server is dead. TIME OUT EXCEEDED"
            client_socket.close()
            return error, ""
        except socket.error as msg:
            error = str(msg)
            error = error + "\n Atlas server is dead."
            self.logger.error(error)
            client_socket.close()
            return error, ""

        client_socket.close()
        data = (list(data))
        ndata = []
        for el1 in zip(data[0::2],data[1::2]):
            ndata.append(chr(el1[0]))
        ndata = ''.join(ndata)
        self.logger.info('Received:'+str(ndata))
        ndata = ndata.replace('\x00', '')
        ndata = ndata.replace("'", "\"")
        connection = json.loads(ndata)
        self.status = connection['Status']
        if(connection['Connection']=='Accepted'):
            self.connected = True
        elif(connection['Connection']=='Denied'):
            self.connected = False
            error = self.status
        return  error, self.status

    def startAtlas(self):
        self.logger.info("Restarting atlas.")
        error = ""
        message = ""
        confirm_start_message = {}
        confirm_start_message["uri"] = "startAtlas"
        a = bitarray('10000011')
        payload = json.dumps(confirm_start_message)
        payload = payload.encode().hex()
        self.calculatePayload((int)(len(payload) * 0.5), a)
        for el in payload:  # introducing security key
            bval = bin(int(el, 16))[2:].zfill(4)
            a.extend(bval)
        error = ''
        self.logger.info('Command sent: START ATLAS')
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            client_socket.connect((self.currentIP, int(self.port)))
            client_socket.sendall(a.tobytes());

            data, header = self.recv_timeout(client_socket);
        except socket.error as msg:
            error = msg
            error = str(error) + "\n In startAtlas: Connection problems with server."
            client_socket.close()
            return error, msg
        client_socket.close()
        self.logger.info("Received:" + str(header[3:].decode('utf-8','ignore').replace("\x00",""))+"-.")
        return error, str(data)

    def cancelSample(self):
        self.logger.info("Killing connections (if any).")
        # establish communication to microscope
        ####################################<<<<<<<<<<<<<<<<
        ## prepare package
        #  FIN 1
        #  RSV1,RSV2,RSV3 000
        #  Opcode 0x8C
        a = bitarray('10001000')
        #  Payload 9 bytes of 0  
        a.extend(9*8*'0')
        log =''
        error = ""
        log = "Sending information to cancel sample:\n"
        self.logger.info('Message sent CANCEL SAMPLE.')
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            client_socket.connect((self.currentIP, int(self.port))) 
            client_socket.sendall(a.tobytes());

            data,header = self.recv_timeout(client_socket);
        except socket.error as msg:
            error = msg
            error = str(error) + "\n In disconnnect :Connection problems with server."
            log = log + str(error)
            client_socket.close()
            return error,str(msg)
        client_socket.close()
        self.logger.info("Received:"+str(header[3:].decode('utf-8','ignore').replace("\x00",""))+"-.")
        return  error,str(data)

    def refresh(self):
        self.logger.info("Refreshing settings.")
        error = ""
        message = ""
        confirm_start_message = {}
        confirm_start_message["uri"] = "refresh"
        a = bitarray('10000011')
        payload = json.dumps(confirm_start_message)
        payload = payload.encode().hex()
        self.calculatePayload((int)(len(payload)*0.5), a)
        for el in payload:
            bval = bin(int(el, 16))[2:].zfill(4)
            a.extend(bval)
        error = ''
        self.logger.info('Command sent: REFRESH')
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            client_socket.connect((self.currentIP, int(self.port)))
            client_socket.sendall(a.tobytes());
            data, header = self.recv_timeout(client_socket);
        except socket.error as msg:
            error = msg
            error = str(error) + "\n In refresh : Connection problems with server."
            client_socket.close()
            return error, []
        client_socket.close()
        self.logger.info("Received:" + str(header[3:].decode('utf-8','ignore').replace("\x00",""))+"-.")
        return error, str(data)

    def initializeSample(self,data_project,workingDir):
        """
            Establishes connection with the microscope and starts one sample
            Comprises 3 messages to the server:
                initialize - information about the project is sent
                setup file - send setup file for starting the project
                start Sample - gives the order of starting
        :param data_project:
        :return:
        """
        error=""
        message=""
        if(not self.connected):
            error = "Not connected to remote server."
            message = []
            return message,error;
        self.logger.info("Initiating sample transmission.")
        data_sample = {}
        data_sample["uri"] ="initializeSample"
        ##
        data_sample["atlas_folder"] =  data_project["atlas_folder"]
        # data_sample["profile"] = data_project["profile"]
        data_sample["setup"] = data_project["setup"]

        data_sample["roi_x_length"] = data_project["roi_x_length"]
        data_sample["roi_y_length"] = data_project["roi_y_length"]
        data_sample["roi_depth"] = data_project["roi_depth"]

        data_sample["dX"] = data_project["dX"]
        data_sample["dY"] = data_project["dY"]
        data_sample["SliceThickness"] = data_project["SliceThickness"]

        data_sample["DefaultStabilizationDuration"] = data_project["DefaultStabilizationDuration"]
        data_sample["AF_Interval"] = data_project["AF_Interval"]
        data_sample["SliceThickness"] = data_project["SliceThickness"]
        data_sample["updatePositionsBeforeStart"] = data_project["has_landmarks"]

        data_sample["x"] = data_project["x"]
        data_sample["y"] = data_project["y"]
        data_sample["z"] = data_project["z"]
        data_sample["tag"] = data_project["tag"]
        data_sample["job_name"] = os.path.split(data_project["atlas_folder"])[1]
        data_sample["unit_scale"] =  data_project["unit_scale"]

        data_sample["cp_x"] = data_project["cp_x"]
        data_sample["cp_y"] = data_project["cp_y"]


        self.scale = data_project["unit_scale"]
        setup_file_folder = workingDir.replace("\\","/")+"/resources/setups/"+data_project["setup"]
        data_sample["local_setup_file"]= setup_file_folder
        self.current_sample = data_sample;
        # establish communication to microscope
        ####################################<<<<<<<<<<<<<<<<
        ## prepare package
        #  FIN 0 - waits for more information, should reply with ACK
        #  RSV1,RSV2,RSV3 000
        #  Opcode 0x3 function for send files and information
        a = bitarray('00000011')
        payload = json.dumps(data_sample)
        payload = payload.encode().hex()
        self.calculatePayload((int)(len(payload)*0.5),a)

        for el in payload: # introducing security key
            bval = bin(int(el, 16))[2:].zfill(4)
            a.extend(bval)
        error = ''
        self.logger.info("Command sent: INITIALIZE")
        self.logger.info(" Sending information of project:"+data_sample["job_name"])
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            client_socket.connect((self.currentIP, int(self.port))) 
            client_socket.sendall(a.tobytes());
            data,header = self.recv_timeout(client_socket,30);
            client_socket.close()
        except socket.error as msg:
            error = str(msg)
            error = error + "In initialize sample : Connection problems with server."
            self.logger.error(error)
            return error,str(message)

        self.logger.info("Received:"+str(header[3:].decode('utf-8','ignore').replace("\x00",""))+"-.")
        if(self.connection_is_accepted(header)):
            ### Prepare header
            ####################################<<<<<<<<<<<<<<<<
            ## prepare package
            #  FIN 1
            #  RSV1,RSV2,RSV3 000
            #  Opcode 0x1 File sending
            a = bitarray('10000001')
            try:
                setup_file_manager = SetupXMLFileManager(setup_file_folder)
            except IOError as msg:
                return "ERROR READING XML SETUP FILE :"+msg,""
            payload = setup_file_manager.getXML()
            if isinstance(payload,str):
                payload = payload.encode('utf-8')
            payload = payload.hex()
            self.calculatePayload(int(len(payload)*0.5),a)
            for el in payload:
                bval = bin(int(el, 16))[2:].zfill(4)
                a.extend(bval)
            ### Now send the 3d_setup file
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                client_socket.connect((self.currentIP, int(self.port)))
                client_socket.sendall(a.tobytes());
                data,header = self.recv_timeout(client_socket,40);
                if(self.connection_is_accepted(header)):
                    time.sleep(2)
                    self.start_ready = True
                    message = "Initialization complete of sample:"+data_sample["tag"]
            except socket.error as msg:
                 error = str(msg)
                 error = error + "\t Connection problems with server."
                 self.logger.error(error)
                 self.start_ready = False
            return error,str(message)

    def startSample(self,data_project):
                error = ""
                message = ""
                confirm_start_message = {}
                confirm_start_message["uri"] ="startSample"
                a = bitarray('10000011')
                payload = json.dumps(confirm_start_message)
                if isinstance(payload, str):
                    payload = payload.encode('utf-8')
                payload = payload.hex()
                self.calculatePayload((int)(len(payload)*0.5),a)
                for el in payload: # introducing security key
                    bval = bin(int(el, 16))[2:].zfill(4)
                    a.extend(bval)
                error = ''
                self.logger.info('Command sent: START SAMPLE')
                client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                try:
                    client_socket.connect((self.currentIP, int(self.port)))
                    client_socket.sendall(a.tobytes());
                    data,header = self.recv_timeout(client_socket,300); # Wait for answer
                    client_socket.close()
                except socket.error as msg:
                    error = str(msg)
                    error = error + "In startSample : Connection problems with server."
                    return error,str(message)

                self.logger.info("Received:"+str(header[3:].decode('utf-8','ignore').replace("\x00",""))+"-.")
                if(self.connection_is_accepted(header)):
                    message = "Sample "+data_project["tag"]+" ready to run."
                    data_project["progress"] = 1
                    error = ''
                else:
                    error = data
                    message = "ERROR"
                    self.logger.error("ERROR :"+str(error))
                return  error,str(message)


    ############## Pinging mechanism #########################
    def startPing(self, iseconds):
        """
        Starts a thread connected to this class by an event.
        The ping is a Timer that continuously (every input seconds) updates
        the application with information about the run.
        It runs in an independent thread that launches an event every
        x seconds.
        :return:
        """
        self.thread_ping = threading.Thread(None,self.cyclicPing,None,[iseconds])
        self.thread_ping.daemon = True
        self.sample_running = True
        self.thread_ping.start()
        self.logger.info("Pinging process started.")

    def stopPing(self):
        self.ping_lock.acquire()
        self.sample_running = False
        self.ping_lock.release()
        self.logger.info("Pinging process stopped.")

    def cyclicPing(self, iseconds):
        inner_condition = True
        while (inner_condition == True):

            time.sleep(iseconds)
            self.ping_lock.acquire()
            inner_condition = self.sample_running
            self.ping_lock.release()
            if(inner_condition == False):
                return

            self.ping()
            self.ping_lock.acquire()
            inner_condition = self.sample_running
            self.ping_lock.release()

    def ping(self):
        """
        Send ping
        If ping is not send the run will continue, but if there is a critical error
        the run stops.
        :return:
        """
        # establish communication to microscope
        ####################################<<<<<<<<<<<<<<<<
        ## prepare package
        #  FIN 0 - waits for more information, should reply with ACK
        #  RSV1,RSV2,RSV3 000
        #  Opcode 0x3 function for send files and information
        a = bitarray('10001001')
        a.extend(9*8*'0') # no payload
        error = ''
        ## Get answer and inform status.
        ### Now send the 3d_setup file
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
                client_socket.connect((self.currentIP, int(self.port)))
                client_socket.sendall(a.tobytes());
                data,header = self.recv_timeout(client_socket,30);
        except socket.error as socket_error:
                 error = str(socket_error) + "\n In ping: Connection problems with server."
                 self.logger.error(error)
                 ## Pop up error message
                 return error
        status = self.eval_status(header)
        ts = time.time()
        message = "" #"At "+datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
        if(status == "OK"):
             # message = message+"ACK"
             # self.logger.info(message)
             return
        else:
             # if warning get more information and inform the use
             if(status == "MESSAGE_READY"):
                error,message = self.getStatus()
                if(message):
                    # Analysis of message to see if there is a problem
                    str_message = message['Message'];
                    result = re.findall('PROMPT', str_message, re.MULTILINE)
                    if(result):
                        pos = str_message.find('PROMPT')
                        self.new_message.emit("M-" + str_message[:pos])
                        self.warning.emit(str_message[pos:])
                        return
                    result = re.findall('REQUEST', str_message, re.MULTILINE)
                    if (result):
                        pos = str_message.find('REQUEST')
                        self.new_message.emit("M-" + str_message[:pos])
                        self.request.emit(str_message[pos:])
                        return
                    result = re.findall('CRITICAL', str_message, re.MULTILINE)
                    if(result):
                        pos = str_message.find('CRITICAL')
                        self.new_message.emit("M-" + str_message[:pos])
                        self.logger.warning(str_message[pos:])
                        self.critical.emit(str_message[pos:])
                        return
                    self.new_message.emit("M-"+str_message)
             elif(status =="CANCELLED"):
                self.critical.emit("CANCELLED")
             elif(status == "PAUSED"):
                self.logger.info("Job paused, waiting to resume.")
             elif(status == "COMPLETED"):
                self.critical.emit("COMPLETED")
             else:
                self.stopPing()
                self.logger.info(str(message))
                self.critical.emit("ERROR."+message)
        return
    
    def getStatus(self):
        error = ""
        message = ""
        status_message = {}
        status_message["uri"] ="getStatus"
        a = bitarray('10000011')
        payload = json.dumps(status_message)
        if isinstance(payload, str):
            payload = payload.encode('utf-8')
        payload = payload.hex()
        self.calculatePayload((int)(len(payload)*0.5),a)

        for el in payload: # introducing security key
            bval = bin(int(el, 16))[2:].zfill(4)
            a.extend(bval)
        self.logger.info('Message sent GET STATUS.')
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            client_socket.connect((self.currentIP, int(self.port)))
            client_socket.sendall(a.tobytes());
            data,header = self.recv_timeout(client_socket,30);
            client_socket.close()
        except socket.error as msg:
            error = str(msg)
            error = error + "In getStatus: Connection problems with server."
            return error,str(message)

        data = data.decode('utf-8','ignore').replace('\x00', '')

        data = data.replace("'", "\"")
        if (self.connection_is_accepted(header)):
            try:
                message = json.loads(str(data))
                error = ''
            except ValueError as e:
                self.logger.error(e)
                error = ""
        else:
            error = json.loads(str(data))
            message = ''
        return  error,message

    def resume(self):
        error = ""
        message = ""
        status_message = {}
        status_message["uri"] = "resume"
        a = bitarray('10000011')
        payload = json.dumps(status_message)
        if isinstance(payload, str):
            payload = payload.encode('utf-8')
        payload = payload.hex()
        self.calculatePayload((int)(len(payload)*0.5), a)

        for el in payload:  # introducing security key
            bval = bin(int(el, 16))[2:].zfill(4)
            a.extend(bval)
        self.logger.info('Message sent RESUME RUN.')
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            client_socket.connect((self.currentIP, int(self.port)))
            client_socket.sendall(a.tobytes());
            data, header = self.recv_timeout(client_socket, 30);
            client_socket.close()
        except socket.error as msg:
            error = str(msg)
            error = error + "In resume: Connection problems with server."
            return error, str(message)

        self.logger.info("Received:" + str(header[3:].decode('utf-8','ignore').replace("\x00",""))+"-.")
        data = data.decode('utf-8','ignore').replace('\x00', '')
        data = data.replace("'", "\"")
        if (self.connection_is_accepted(header)):
            return ''
        else:
            error = json.loads(str(data))
            return error

    def pushBC(self):
        error = ""
        message = ""
        status_message = {}
        status_message["uri"] = "pushBC"
        a = bitarray('10000011')
        payload = json.dumps(status_message)
        if isinstance(payload, str):
            payload = payload.encode('utf-8')
        payload = payload.hex()
        self.calculatePayload((int)(len(payload)*0.5), a)
        for el in payload:  # introducing security key
            bval = bin(int(el, 16))[2:].zfill(4)
            a.extend(bval)
        self.logger.info('Message sent PUSH CURRENT Brightness and CONTRAST.')
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            client_socket.connect((self.currentIP, int(self.port)))
            client_socket.sendall(a.tobytes());
            data, header = self.recv_timeout(client_socket, 30);
            client_socket.close()
        except socket.error as msg:
            error = str(msg)
            error = error + "In resume: Connection problems with server."
            return error, str(message)

        self.logger.info("Received:" + str(header[3:].decode('utf-8','ignore').replace("\x00",""))+"-.")
        data = data.decode('utf-8','ignore').replace('\x00', '')
        data = data.replace("'", "\"")
        if (self.connection_is_accepted(header)):
            return ''
        else:
            error = json.loads(str(data))
            return error



    def resume_server_request(self):
        error = ""
        message = ""
        status_message = {}
        status_message["uri"] = "resume_server_request"
        a = bitarray('10000011')
        payload = json.dumps(status_message)
        if isinstance(payload, str):
            payload = payload.encode('utf-8')
        payload = payload.hex()
        self.calculatePayload((int)(len(payload)*0.5), a)
        for el in payload:  # introducing security key
            bval = bin(int(el, 16))[2:].zfill(4)
            a.extend(bval)
        self.logger.info('Message sent RESUME SERVER REQUEST.')
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            client_socket.connect((self.currentIP, int(self.port)))
            client_socket.sendall(a.tobytes());
            data, header = self.recv_timeout(client_socket, 30);
            client_socket.close()
        except socket.error as msg:
            error = str(msg)
            error = error + "In RESUME SERVER REQUEST: Connection problems with server."
            return error, ""

        self.logger.info("Received:" +str(header[3:].decode('utf-8','ignore').replace("\x00",""))+"-.")
        data = data.decode('utf-8','ignore').replace('\x00', '')
        data = data.replace("'", "\"")
        if (self.connection_is_accepted(header)):
            return ''
        else:
            error = json.loads(str(data))
            return error

    def eval_status(self,header):

        if(header[0]==154): #
            return "MESSAGE_READY"
        elif (header[0] == 170):# 0xAA
            return "INITIALIZED"
        elif(header[0]==186): # 0xBA
            return "CANCELLED"
        elif(header[0] == 202): # 0xCA
            return "PAUSED"
        elif (header[0] == 218):  # 0xDA
            return "IDLE"
        elif (header[0]==234): #0xEA
            return "COMPLETED"
        elif (header[0] == 250):  # 0xFA
            return "ERROR"
        else:
            return "OK" ##

    def calculatePayload(self,payload_length,a):
        """

        :param self:
        :param payload_length:
        :param a: bytearray
        :return:
        """

        if(payload_length>0): # there is payload, we need to add more data
              if(payload_length)<126:
                 to_add = format(payload_length,'#010b')
                 a.extend(to_add[2:])
                 a.extend(8*8*'0')
              elif(payload_length > 126 and payload_length<65536):
                 a.extend('01111110') #126 average
                 to_add = format(payload_length,'#018b')
                 for i in range(len(to_add),2,-8):
                     a.extend(to_add[i-8:i])
                 a.extend(8*6*'0')
              else:
                 a.extend('01111111') #127 very big
                 to_add = format(payload_length,'#066b')
                 for i in range(len(to_add),2,-8):
                     a.extend(to_add[i-8:i])
        return

    def connection_is_accepted(self,header):
        if(header[0] == 140):
            return True
        return False

    def recv_timeout(self,my_socket, timeout=2):
      """

      :param self:
      :param my_socket: socket to evaluate
      :param timeout: maximum waiting
      :return: error message decoded
      """
      #make socket non blocking
      my_socket.setblocking(0)
     
      #total data partwise in an array
      total_data=[];
      data='';
      header='';

      #beginning time
      begin=time.time()
      while 1:
        #recv something
        try:
            buf = bytearray(32)
            nbytes = my_socket.recv_into(buf, 32)
            if nbytes > 0 :
                total_data.append(buf)
                #change the beginning time for measurement
                begin=time.time()
                if  nbytes>=32:
                    break
            else:
                #sleep for sometime to indicate a gap
                time.sleep(0.1)
        except:
            pass
        #if you got some data, then break after timeout

        waiting_time = time.time()-begin
        #if you got no data at all, wait a little longer, twice the timeout
        if waiting_time > timeout*2:
            raise socket.timeout
            break

      header = total_data[0]
      payload_size = int(header[1])
      if(payload_size>0): # there is payload, we need to get more data
              if(payload_size<126):
                  payload = payload_size
              elif(payload_size == 126):
                  payload_len = header[2:4]
                  payload_len.reverse()
                  payload_len = list(payload_len)
                  nlist = []
                  for el in payload_len:
                    nlist.append('{0:08b}'.format(el))
                  nlist = ''.join(nlist)
                  payload = int(nlist,2)
              elif(payload_size == 127):
                  payload_len = header[2:10]
                  payload_len.reverse()
                  payload_len = list(payload_len)
                  nlist = []
                  for el in payload_len:
                    nlist.append('{0:08b}'.format(el))
                  nlist = ''.join(nlist)
                  payload = int(nlist,2)
      else:
          return [],header
      
      #beginning time
      begin=time.time()
    
      total_data = bytearray(); # Should hold until 536,870,912
      total_bytes = 0;
      while 1:
        #if you got some data, then break after timeout
        if total_bytes >= payload:
            break
        #if you got no data at all, wait a little longer, twice the timeout
        elif time.time()-begin > timeout*2:
            break
         
        #recv something
        try:
            buf = bytearray(1024)
            nbytes = my_socket.recv_into(buf,1024)  # we will get packets of 1024
            if nbytes > 0:
                total_data.extend(buf)
                total_bytes = total_bytes + nbytes
                #change the beginning time for measurement
                begin=time.time()
            else:
                #sleep for sometime to indicate a gap
                time.sleep(0.1)
        except:
            pass
      #join all parts to make final string
      return total_data,header

    def getCurrentStagePosition(self):
        error = ""
        message = ""
        status_message = {}
        status_message["uri"] ="getStagePositionXYZ"
        a = bitarray('10000011')
        payload = json.dumps(status_message)
        if isinstance(payload, str):
            payload = payload.encode('utf-8')
        payload = payload.hex()
        self.calculatePayload((int)(len(payload)*0.5),a)
        for el in payload: # introducing security key
            bval = bin(int(el, 16))[2:].zfill(4)
            a.extend(bval)
        self.logger.info('Message sent GET CURRENT STAGE POSITION.')
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            client_socket.connect((self.currentIP, int(self.port)))
            client_socket.sendall(a.tobytes());
            data,header = self.recv_timeout(client_socket,30);
            client_socket.close()
        except socket.error as msg:
            error = str(msg)
            error = error + "In getCurrentStatePosition: Connection problems with server."
            return error,""
        data = data.decode('utf-8','ignore').replace('\x00', '')
        data = data.replace("'", "\"")
        log = "Received:"+str(header[3:].decode('utf-8','ignore').replace("\x00",""))+"-."
        if(self.connection_is_accepted(header)):
            jcoord = json.loads(str(data))
            coord = [0,0,0]
            coord[0] = float(jcoord['xpos'].replace(',','.'))
            coord[1] = float(jcoord['ypos'].replace(',','.'))
            coord[2] = float(jcoord['zpos'].replace(',','.'))
            if(coord[0]<0):
                coord[0]=-coord[0]
            if(coord[1]<0):
                coord[1]=-coord[1]
            coord = np.array(coord)
            error = ''
        else:
            error = data
        return  error,coord

    def setStageXYPosition(self,coord):
        error = ""
        message = ""
        status_message = {}
        status_message["scale"] = self.scale
        status_message["uri"] ="setStagePositionXYZ"
        status_message["xpos"] = str(coord[0])
        status_message["ypos"] = str(coord[1])
        if(len(coord)<3):
            status_message["zpos"] = 0.0
        else:
            status_message["zpos"] = str(coord[2])
        a = bitarray('10000011')
        payload = json.dumps(status_message)
        if isinstance(payload, str):
            payload = payload.encode('utf-8')
        payload = payload.hex()
        self.calculatePayload((int)(len(payload)*0.5),a)
        for el in payload: # introducing security key
            bval = bin(int(el, 16))[2:].zfill(4)
            a.extend(bval)
        self.logger.info('Message sent SET STAGE POSITION XY.')
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            client_socket.connect((self.currentIP, int(self.port)))
            client_socket.sendall(a.tobytes());
            data,header = self.recv_timeout(client_socket,30);
            client_socket.close()
        except socket.error as msg:
            error = str(msg)
            error = error + "In setStageXYPosition: Connection problems with server."
            return error,""

        log = "Received:"+str(header[3:].decode('utf-8','ignore').replace("\x00",""))
        if(self.connection_is_accepted(header)):
            message = "Moving to position:"+str(coord)
            error = ''
        else:
            error = data
        return  error,str(message)

    def updateJobStagePositionXY(self, coord):
        error = ""
        message = ""
        status_message = {}
        status_message["scale"] = self.scale
        status_message["uri"] = "updateJobStagePositionXY"
        status_message["xpos"] = str(coord[0])
        status_message["ypos"] = str(coord[1])
        if (len(coord) < 3):
            status_message["zpos"] = 0.0
        else:
            status_message["zpos"] = str(coord[2])
        a = bitarray('10000011')
        payload = json.dumps(status_message)
        if isinstance(payload, str):
            payload = payload.encode('utf-8')
        payload = payload.hex()
        self.calculatePayload((int)(len(payload) * 0.5), a)

        for el in payload:  # introducing security key
            bval = bin(int(el, 16))[2:].zfill(4)
            a.extend(bval)
        self.logger.info('Message sent Update Job Stage Position XY.')
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            client_socket.connect((self.currentIP, int(self.port)))
            client_socket.sendall(a.tobytes());
            data, header = self.recv_timeout(client_socket, 30);
            client_socket.close()
        except socket.error as msg:
            error = str(msg)
            error = error + "In updateJobStagePositionXY: Connection problems with server."
            return error, ""

        log = "Received:" + str(header[3:].decode('utf-8', 'ignore').replace("\x00", ""))
        if (self.connection_is_accepted(header)):
            message = "Job position updated to:" + str(coord)
            error = ''
        else:
            error = data
        return error, str(message)


    def grabImage(self,dwell_time,pixel_size,resolution,line_average,scan_rotation,pathImages,name,shared=True):
        error = ""
        message = ""
        status_message = {}
        status_message["uri"] ="grabFrame"
        if(shared):
            status_message["tag"] = name
            status_message["in_shared_folder"] = "True"
            status_message["shared_folder"] = pathImages
        else:
            status_message["tag"] = name
            status_message["in_shared_folder"] = "False"

        status_message["dwell_time"] = dwell_time # 5.0
        status_message["pixel_size"] = pixel_size #0.8  # microns
        status_message["resolution"] = resolution #6 # Grided square
        status_message["line_average"] = line_average #3
        status_message["scan_rotation"] = scan_rotation

        a = bitarray('10000011')
        payload = json.dumps(status_message)
        if isinstance(payload, str):
            payload = payload.encode('utf-8')
        payload = payload.hex()
        self.calculatePayload((int)(len(payload)*0.5),a)

        for el in payload: # introducing security key
            bval = bin(int(el, 16))[2:].zfill(4)
            a.extend(bval)
        self.logger.info('Message sent GRAB IMAGE. '+name);
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            client_socket.connect((self.currentIP, int(self.port)))
            client_socket.sendall(a.tobytes());
            data,header = self.recv_timeout(client_socket,120);
            client_socket.close()
        except socket.error as msg:
            error = str(msg)
            error = error + "In grabImage: Connection problems with server."
            return error,""

        log = "Received:"+str(header[3:].decode('utf-8','ignore').replace("\x00",""))
        if(self.connection_is_accepted(header)):
            message = "OK"
            error = ''
            data = data.decode('utf-8','ignore').replace('\x00', '')
            data = data.replace("'", "\"")
            try:
                answer = json.loads(str(data))
            except ValueError as e:
                self.logger.error(e)
                return

            filename = answer['filename']+".tif"
            filename = pathImages +"\\"+ filename
            if(not shared): # data needs to be parsed to a tiff file)
                recv_image = answer['imagefile']
                recv_image = base64.b64decode(recv_image)
                with open(filename, 'wb') as f:
                    f.write(recv_image)
        else:
            error = data
        return  error,filename

    def autoFocusSurface(self, typeAFAS = "SURFACE"):

        if  typeAFAS =="EMERGENCY":
            uri = "autoFocusEmergency"
        else:
            uri = "autoFocusSurface"

        error = ""
        message = ""
        status_message = {}
        status_message["uri"] = uri
        a = bitarray('10000011')
        payload = json.dumps(status_message)
        if isinstance(payload, str):
            payload = payload.encode('utf-8')
        payload = payload.hex()
        self.calculatePayload((int)(len(payload)*0.5),a)

        for el in payload: # introducing security key
            bval = bin(int(el, 16))[2:].zfill(4)
            a.extend(bval)
        self.logger.info('Message sent AUTOFOCUS.')
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            client_socket.connect((self.currentIP, int(self.port)))
            client_socket.sendall(a.tobytes());
            data,header = self.recv_timeout(client_socket,120);
            client_socket.close()
        except socket.error as msg:
            error = str(msg)
            error = error + "In AFSurface: Connection problems with server."
            return error,str(message)
        data = data.decode('utf-8','ignore').replace('\x00', '')
        data = data.replace("'", "\"")
        log = "Received:"+str(header[3:].decode('utf-8','ignore').replace("\x00",""))
        if(self.connection_is_accepted(header)):
            self.logger.info("Function "+uri+" executed")
            error = ''
            data = data.replace('\x00', '')
            data = data.replace("'", "\"")
            answer = json.loads(str(data))
            self.logger.info(str(answer))
        else:
            error = data
        return  error



    def setValue(self, command, value):
        error = ""

        status_message = {}
        status_message["uri"] = "SmartSEMcmd"
        status_message["command"] = command
        status_message["value"] = value

        a = bitarray('10000011')
        payload = json.dumps(status_message)
        if isinstance(payload, str):
            payload = payload.encode('utf-8')
        payload = payload.hex()
        self.calculatePayload((int)(len(payload)*0.5), a)

        for el in payload:  # introducing security key
            bval = bin(int(el, 16))[2:].zfill(4)
            a.extend(bval)
        self.logger.info('Message sent SET VALUE :' + command)
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            client_socket.connect((self.currentIP, int(self.port)))
            client_socket.sendall(a.tobytes());
            data, header = self.recv_timeout(client_socket, 30);
            client_socket.close()
        except socket.error as msg:
            error = str(msg)
            error = error + "In setValue: Connection problems with server."
            return error

        log = "Received:" + str(header[3:].decode('utf-8','ignore').replace("\x00",""))+"-."
        if (self.connection_is_accepted(header)):
            error = "Command succesfully executed:" + command + " :" + str(value)
        else:
            error = data
        return error

    def pause(self):
        error = ""
        message = ""
        cancel_message = {}
        cancel_message["uri"] = "pause"
        a = bitarray('10000011')
        payload = json.dumps(cancel_message)
        if isinstance(payload, str):
            payload = payload.encode('utf-8')
        payload = payload.hex()
        self.calculatePayload((int)(len(payload)*0.5), a)
        for el in payload:  # introducing security key
            bval = bin(int(el, 16))[2:].zfill(4)
            a.extend(bval)
            error = ''
        self.logger.info('Message sent PAUSE RUN.')
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            client_socket.connect((self.currentIP, int(self.port)))
            client_socket.sendall(a.tobytes());
            data, header = self.recv_timeout(client_socket, 30);
            client_socket.close()
        except socket.error as msg:
            error = str(msg)
            error = error + "In pause: Connection problems with server."
            return error, ""

        log = "Received:" + str(header[3:].decode('utf-8','ignore').replace("\x00",""))+"-."
        if (self.connection_is_accepted(header)):
            message = "Sample paused."
            error = ''
        else:
            message = ''
            error = data
        return error, message


    def getMasterFolder(self):
        error = ""
        message = ""
        cancel_message = {}
        cancel_message["uri"] = "getMasterFolder"
        a = bitarray('10000011')
        payload = json.dumps(cancel_message)
        if isinstance(payload, str):
            payload = payload.encode('utf-8')
        payload = payload.hex()
        self.calculatePayload((int)(len(payload)*0.5), a)
        for el in payload:  # introducing security key
            bval = bin(int(el, 16))[2:].zfill(4)
            a.extend(bval)
            error = ''
        self.logger.info('Message sent obtain main Folder.')
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            client_socket.connect((self.currentIP, int(self.port)))
            client_socket.sendall(a.tobytes());
            data, header = self.recv_timeout(client_socket, 30);
            client_socket.close()
        except socket.error as msg:
            error = str(msg)
            error = error + "In pause: Connection problems with server."
            return error, ""

        log = "Received:" + str(header[3:].decode('utf-8','ignore').replace("\x00",""))+"-."
        if (self.connection_is_accepted(header)):
            message = "OK"
            error = ''
            data = data.decode('utf-8', 'ignore').replace('\x00', '')
            data = data.replace("'", "\"")
            try:
                answer = json.loads(str(data))
            except ValueError as e:
                self.logger.error(e)
                return
            message = answer['folder_name']
        else:
            message = ''
            error = data
        return error, message

    #### New way of implementing calls to server
    def setAutoCP(self, master_folder):
        status_message = {}
        status_message["unit_scale"] = self.scale
        status_message["uri"] ="autoCP"
        status_message["shared_folder"] = master_folder
        status_message["cp_x"] = "-50.0"
        status_message["cp_y"] = "0.0"
        return self.sendMessageFunction(status_message, "Message sent Automatic CP")


    def makeTrench(self, master_folder, address_setup_path):
        status_message = {}
        status_message["unit_scale"] = self.scale
        status_message["uri"] ="makeTrenchRecipe"
        status_message["shared_folder"] = master_folder
        status_message["profile_path"] = address_setup_path
        return self.sendMessageFunction(status_message, "Message sent Make a trench")


    def sendMessageFunction(self, status_message, output_message):
        error = ""
        message = ""
        a = bitarray('10000011')
        payload = json.dumps(status_message)
        if isinstance(payload, str):
            payload = payload.encode('utf-8')
        payload = payload.hex()
        self.calculatePayload((int)(len(payload)*0.5),a)

        for el in payload: # introducing security key
            bval = bin(int(el, 16))[2:].zfill(4)
            a.extend(bval)
        self.logger.info(output_message)
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            client_socket.connect((self.currentIP, int(self.port)))
            client_socket.sendall(a.tobytes())
            data,header = self.recv_timeout(client_socket,30)
            client_socket.close()
        except socket.error as msg:
            error = str(msg)
            error = error + "Connection problems with server."
            return error,""

        message = "Received:"+str(header[3:].decode('utf-8','ignore').replace("\x00",""))
        if(self.connection_is_accepted(header)):
            self.logger.info(message)
            self.logger.info('Function'+ status_message["uri"]+' executed.')
            error = ''
        else:
            error = data
        return error,str(message)