import smtplib
import numpy as np
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# set up the SMTP server
s = smtplib.SMTP(host = 'smtp.gmail.com', port = 587)
s.starttls()
s.login('smartsurveillancehc@gmail.com', 'VVdu6K3ym7Pi6Jp')

warningTexts = []
warningTexts.append('Please check on the observed person since a fall was detected by the Smart Surveillance Home Caring Device')
warningTexts.append('Please check on the observed person since an intruder was detected by the Smart Surveillance Home Caring Device')
warningTexts.append('Please check on the observed person since they were in a bad mood for a long time.')

def sendMsg(warningIndicator):
	

	msg = MIMEMultipart()

	msg['From'] = 'smartsurveillancehc@gmail.com'
	msg['To'] = 'bastianklopfer384@gmail.com,chaoyan0411@gmail.com'
	msg['Subject'] = "WARNING MESSAGE"

	msg.attach(MIMEText(warningTexts[warningIndicator], 'plain'))

	s.send_message(msg)

	del msg

