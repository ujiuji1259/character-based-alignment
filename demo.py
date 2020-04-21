from jsc import JSC
from flask import Flask, render_template, request
import argparse

app = Flask(__name__)

@app.route("/", methods=['POST'])
def index():
    word = request.form["text"]
    normalized = jsc.decode(word)[1].replace(' ', '')
    return normalized

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build model files')
    parser.add_argument('--vocab', type=str, help="specify vocab file")
    parser.add_argument('--train', type=str, help="specify vocab file")
    parser.add_argument('--ngram', type=int, help="specify ngram length")
    args = parser.parse_args()
    jsc = JSC(args.vocab, args.ngram)
    jsc.load_trained_file()
    print('finish_load')
    jsc.create_dst_list(args.train)
    app.run(port='8000', host='0.0.0.0', debug=True)
