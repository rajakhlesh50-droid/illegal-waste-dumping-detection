with open("app.py", "rb") as f:
    code = f.read().decode("utf-8")

if "import re" not in code:
    code = code.replace("import csv\n", "import csv\nimport re\n", 1)
    with open("app.py", "wb") as f:
        f.write(code.encode("utf-8"))
    print("Added import re to app.py!")
else:
    print("import re already present")

