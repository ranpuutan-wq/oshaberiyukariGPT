# tools/list_sapi_voices.py
import win32com.client
sp = win32com.client.Dispatch("SAPI.SpVoice")
vs = sp.GetVoices()
for i in range(vs.Count):
    v = vs.Item(i)
    try:
        print(i, v.GetDescription())
    except Exception:
        print(i, "<error>")
        