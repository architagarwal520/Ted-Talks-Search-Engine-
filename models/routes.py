
from flask import Flask, render_template, request,redirect,url_for
from models import app
from models.nav_test import get_cosine





@app.route("/",methods=['POST','GET']) 
def home():
	if request.method=="POST":
		search_talk=request.form["query"]
		urls=get_cosine(search_talk)
		for url in urls:
			url= url + "&output=embed"
		return render_template('search.html',urls=urls)
	return render_template('search.html')