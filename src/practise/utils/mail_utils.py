import mailtrap as mt
from practise.config import MAILTRAP_API_KEY

def send_appointment_email(to_email: str, to_name: str, date: str, time: str):
    mail = mt.Mail(
        sender=mt.Address(email="hello@demomailtrap.co", name="Mailtrap Test"),
        to=[mt.Address(email=to_email, name=to_name)],
        subject="Your Appointment Details!",
        text=f"Hello {to_name},\nYour appointment is scheduled on {date} at {time}.\nThank you!",
        category="Appointment Notification",
    )
    client = mt.MailtrapClient(token=MAILTRAP_API_KEY)
    return client.send(mail)
