import llm_client
client = llm_client.Client(address="babel.lti.cs.cmu.edu", port = 8080)
i =0
while(i<10):
    outputs = client.prompt("CMU's PhD students are")
    print(outputs[0].text)
    i+=1
