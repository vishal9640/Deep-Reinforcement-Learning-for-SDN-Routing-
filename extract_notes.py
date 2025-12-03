import zipfile
import xml.etree.ElementTree as ET
import os
p = os.path.join(os.path.dirname(__file__), 'Notes.docx')
if not os.path.exists(p):
    print('Notes.docx not found at', p)
    raise SystemExit(1)
with zipfile.ZipFile(p) as z:
    try:
        xml = z.read('word/document.xml')
    except KeyError:
        print('document.xml not found inside docx')
        raise
root = ET.fromstring(xml)
ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
texts = []
for para in root.findall('.//w:p', ns):
    parts = [t.text for t in para.findall('.//w:t', ns) if t.text]
    if parts:
        texts.append(''.join(parts))
print('\n\n'.join(texts))
