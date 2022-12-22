import json
import boto3

def hello(event, context):
    client=boto3.client('ssm')
    # response = client.get_parameter(Name='gajraj')
    # print(f"Data --> {response['Parameter']['Value']}")
    response = client.get_parameter(Name='gpassword',WithDecryption=True)
    print(f"Data --> {response['Parameter']['Value']}")
    body = {
        "message": "Go Serverless v3.0! Your function executed successfully!"
    }

    return {"statusCode": 200, "body": json.dumps(body)}
