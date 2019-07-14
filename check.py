import sentiment_mod as s

print(s.sentiment("This movie was awesome! The acting was great, plot was wonderful, and there were pythons...so yea!"))
print(s.sentiment("This movie was utter junk. There were absolutely 0 pythons. I don't see what the point was at all. Horrible movie, 0/10"))

while(True):
	sentence = input("Enter your sentence: ")
	print("result:" + str(s.sentiment(sentence)[0]) + " " + str(s.sentiment(sentence)[1]*100) + "%")