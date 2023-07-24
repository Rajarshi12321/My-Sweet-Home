from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('home.html', new_variable=None)

@app.route('/process_option', methods=['POST'])
def process_option():
    selected_option = request.form['selected_option']
    # You can process the selected_option here and generate a new variable
    new_variable = selected_option.upper()  # Example: Convert to uppercase
    return render_template('home.html', new_variable=new_variable)

if __name__ == "__main__":
    app.run(debug=True)
