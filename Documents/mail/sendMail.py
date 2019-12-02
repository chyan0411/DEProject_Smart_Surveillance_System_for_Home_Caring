import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# set up the SMTP server
s = smtplib.SMTP(host = 'smtp.gmail.com', port = 587)
s.starttls()
s.login('smartsurveillancehc@gmail.com', 'VVdu6K3ym7Pi6Jp')

msg = MIMEMultipart()

msg['From'] = 'smartsurveillancehc@gmail.com'
msg['To'] = 'bastianklopfer384@gmail.com,chaoyan0411@gmail.com'
msg['Subject'] = "TEST"

msg.attach(MIMEText('Bastian ist dumm', 'plain'))

s.send_message(msg)

del msg

s.quit()
