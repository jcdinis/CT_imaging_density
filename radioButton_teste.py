import wx

class MyForm(wx.Frame):

    #----------------------------------------------------------------------
    def __init__(self):
        wx.Frame.__init__(self, None, wx.ID_ANY, "Tutorial")
        panel = wx.Panel(self, wx.ID_ANY)

        self.radio = wx.RadioButton(panel, label="Test", style = wx.RB_GROUP)
        self.radio2 = wx.RadioButton(panel, label="Test2")
        self.radio3 = wx.RadioButton(panel, label="Test3")

        btn = wx.Button(panel, label="Check Radio")
        btn.Bind(wx.EVT_BUTTON, self.onBtn)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.radio, 0, wx.ALL, 5)
        sizer.Add(self.radio2, 0, wx.ALL, 5)
        sizer.Add(self.radio3, 0, wx.ALL, 5)
        sizer.Add(btn, 0, wx.ALL, 5)
        panel.SetSizer(sizer)

    #----------------------------------------------------------------------
    def onBtn(self, event):
        """"""
        print "First radioBtn = ", self.radio.GetValue()
        print "Second radioBtn = ", self.radio2.GetValue()

# Run the program
if __name__ == "__main__":
    app = wx.PySimpleApp()
    frame = MyForm().Show()
    app.MainLoop()
