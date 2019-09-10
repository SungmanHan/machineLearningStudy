# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 00:05:57 2019

@author: powergen
"""

import pandas as pd

readExcelData = pd.read_excel('company_list.xlsx')

rowCount = readExcelData['이메일'].count()

if rowCount > 0 :
  for i, row in readExcelData.iterrows():
    print(row['상호'], row['담당자'], row['이메일'])
  
    import win32com.client
    from win32com.client import Dispatch, constants
    
    const=win32com.client.constants
    olMailItem = 0x0
    obj = win32com.client.Dispatch("Outlook.Application")
    newMail = obj.CreateItem(olMailItem)
    newMail.Subject = "I AM SUBJECT!!"
    newMail.Body = "I AM\nTHE BODY MESSAGE!"
    #newMail.BodyFormat = 2 # olFormatHTML https://msdn.microsoft.com/en-us/library/office/aa219371(v=office.11).aspx
    #newMail.HTMLBody = "<HTML><BODY>Enter the <span style='color:red'>message</span> text here.</BODY></HTML>"
    newMail.To = "colt95632@naver.com"
    #attachment1 = r"C:\Temp\example.pdf"
    #newMail.Attachments.Add(Source=attachment1)
    newMail.Display()
    newMail.Send()


from tkinter import *
from tkinter import ttk
from tkinter import messagebox
win = Tk ()
win.title("Raspberry Pi UI")
win.geometry('200x100+200+200')
#def clickMe():
#  messagebox.showinfo("Button Clicked", str.get())
str = StringVar()
textbox = ttk.Entry(win, width=20, textvariable=str)
textbox.grid(column = 0 , row = 0)
action=ttk.Button(win, text="Click Me", command=clickMe)
action.grid(column=0, row=1)
print(str.get())
win.mainloop()


