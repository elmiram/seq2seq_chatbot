import telebot
from telebot import types
from seq2seqbot import conf
import tensorflow as tf
import execute
import re

regPunct = re.compile(' +([.,?!":;\'\]\)\n])')
regSpace = re.compile('\s+')
sess = tf.Session()
sess, model, enc_vocab, rev_dec_vocab = execute.init_session(sess, conf='../seq2seq_serve.ini')
print('created sess')
bot = telebot.TeleBot(conf.token)
print('ready')


@bot.message_handler(commands=['start'])
def start(msg):
    bot.send_message(msg.chat.id, 'привет!')


def normalize(text):
    text = text.replace('_UNK', '')
    text = regPunct.sub('\\1', text)
    text = regSpace.sub(' ', text)
    if not text.strip():
        text = '¯\_(ツ)_/¯'
    return text.strip()


@bot.message_handler(content_types=["text"])
def answer(message):
    reply = normalize(execute.decode_line(sess, model, enc_vocab, rev_dec_vocab, message.text))
    reply += ' ' + normalize(execute.decode_line(sess, model, enc_vocab, rev_dec_vocab, reply))
    bot.send_message(message.chat.id, reply)


bot.polling(none_stop=True)


