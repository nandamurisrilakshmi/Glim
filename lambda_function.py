import boto3
import json
import PyPDF2 # pip install PyPDF2
import docx   # pip install python-docx


rek_client = boto3.client('rekognition')
comp_client = boto3.client('comprehend')
s3_client = boto3.client('s3')



def lambda_handler(event, context):
    
    records = [x for x in event.get('Records', []) if x.get('eventName') == 'ObjectCreated:Put']
    sorted_events = sorted(records, key=lambda e: e.get('eventTime'))
    latest_event = sorted_events[-1] if sorted_events else {}
    info = latest_event.get('s3', {})
    file_key = info.get('object', {}).get('key')
    bucket_name = info.get('bucket', {}).get('name')
    file_path = "{}".format(file_key)

    extension =  file_key.split(".")[-1]
    
    plain_text = ''
    if extension == "pdf":
        plain_text = pdf_text(file_path)
        data = comprehend(plain_text)
    if extension == "docx":
        plain_text = docx_text(file_path)
        data = comprehend(plain_text)
    if extension == "txt":
        plain_text = txt_text(file_path)
        data = comprehend(plain_text)
    if extension == "jpg" or extension == "jpeg" or extension == "png":
       data = rekognition(bucket_name,file_key)
    key=file_key.split(".")[0]+'glim.txt'
    s3_client.put_object(Body=data, Bucket='glim-output', Key=key)
    
    return "ok"

def rekognition(bucket,key):
    image_nudity=[]
    image_text=[]

    response = rek_client.detect_moderation_labels({"Image": {"S3Object": {"Bucket": bucket,"Name": key}},"MinConfidence": 60})
    for l in response['ModerationLabels']:
        image_nudity.append(l['Name'])
    response=rek_client.detect_text(Image={'S3Object':{'Bucket':bucket,'Name':key}})
    textDetections=response['TextDetections']
    for text in textDetections:
        image_text.append(text['DetectedText'])
    text = ' '.join(image_text)
    text_analysis=comprehend(text)
    image_data = image_nudity+image_text+text_analysis
    image_data = ' '.join(image_data)
    return image_data

def comprehend(text):

    
    dominant_language = 'en'
          
    response = comp_client.detect_entities(
        Text=text,
        LanguageCode=dominant_language
    )
    entites = list(set([x['Type'] for x in response['Entities']]))
    result_entities = ' '.join(entities)
    response = comp_client.detect_sentiment(
        Text=text,
        LanguageCode=dominant_language
    )
    senti=response['Sentiment']
    result = entites+senti
    result  = ' '.join(result)
    return result
    
    
def pdf_text(file_path):
    pdf_file_obj = open(file_path,'rb')
    pdf_reader = PyPDF2.PdfFileReader(pdf_file_obj)
    tot_pages = pdf_reader.numPages
    text = []
    for i in range(tot_pages):
        pageObj = pdf_reader.getPage(i)
        text.append(pageObj.extractText())
        
    return "\n".join(text)
    
def docx_text(file_path):
    doc = docx.Document(file_path)
    all_text = []
    for doc_para in doc.paragraphs:
        all_text.append(doc_para.text)
    return "\n".join(all_text)

def txt_text(file_path):
    with open(file_path, "r") as f:
        text = f.read()
    return text
