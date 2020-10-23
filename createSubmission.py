import os, glob, shutil, sys
from zipfile import ZipFile, ZIP_DEFLATED

EXINDEX = 3

def zipdir(path, ziph):
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file))

def cleanString(string):
	return str.lower(''.join(e for e in string if e.isalnum()))

if __name__ == '__main__':
	print("Creating exercise .zip")
	pdfFiles = glob.glob("*.pdf")
	maxIndex = len(pdfFiles)-1
	for (i, file) in enumerate(pdfFiles):
		print(f' {i}) {file}')
	pdfIndex = int(input(f"Which .PDF file is your hand-in (enter integer index 0-{maxIndex})? >>> "))
	if pdfIndex > maxIndex or pdfIndex < 0:
		raise Exception("Out of range integer chosen!")
	elif not(os.path.exists("code") and os.path.isdir("code")):
		raise Exception("code directory does not exist!")
	else:
		expdforig = pdfFiles[pdfIndex]
		print(f" - File selected: {expdforig}")
		user1 = input("Group member 1: >>> ")
		user2 = input("Group member 2: >>> ")
		user1 = cleanString(user1)
		user2 = cleanString(user2)
		print(f"Exercise .zip file being created for the group members '{user1}' and '{user2}'")
		exname = f"PR-EX_{EXINDEX}_{user1}_{user2}"
		expdf = exname+'.pdf'
		exzip = exname+'.zip'
		try:
			os.mkdir(exname)
			print(f" - Temporary dir created: {exname}")
			shutil.copy(expdforig, os.path.join(exname,expdf))
			print(f" - Copied exercise pdf file: {expdforig}")
			shutil.copytree("code", os.path.join(exname, "code"))
			print(" - Copied the code folder")
			zipf = ZipFile(exzip, 'w', ZIP_DEFLATED)
			zipdir(exname, zipf)
			zipf.close()
			print(f" - .zip file for upload created: {exzip}")
			shutil.rmtree(exname)
			print(" - Temporary dir removed")
		except:
			raise(f"Error occured while creating submission .zip: {sys.exc_info()[0]}")