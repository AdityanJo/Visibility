import tensorflow as tf
import numpy as np
import cv2
import argparse
from config import Config
from model import CaptionGenerator
from misc import CaptionData, TopN
from dataset import prepare_test_data
from tqdm import tqdm
import time
import pickle
import os
# import wget
# import nltk
# nltk.download('punkt')

class DeepRNNInference(object):
	# @class_method
	# def from_path(cls, model_dir):
	# 	models=[]
	# 	for i in cls.columns:
	# 		models.append(pickle.load(open(model_dir+'')))

	@classmethod
	def from_path(cls, model_dir):
		# models=[]
		# weight_files=os.listdir(model_dir)
		# weight_files=[wf for wf in weight_files if wf.endswith('.npy')]
		# for weight_file in weight_files:
		# config=Config()
		# config.phase='test'
		# config.train_cnn=False
		# config.beam_size=3
		# config.batch_size=1
		# sess = tf.Session()
		# sess.__enter__()
		# model=CaptionGenerator(config)
		# sess.run(tf.global_variables_initializer())
		# model.load(sess,model_dir+'/289999.npy')
		return cls(model_dir+'/289999.npy')

	def __init__(self, weight_file, beam_size=5,  save_to='test.png', mean_file='ilsvrc_2012_mean.npy'):
		# self.image=self.load_image(image_file)
		# url='https://vision.ece.vt.edu/mscoco/downloads/captions_train2014.json'
		# wget.download(url,out='.')
		# self.mean=np.load(mean_file).mean(1).mean(1)
		self.mean=np.array([104.00698793, 116.66876762, 122.67891434])
		self.scale_shape=np.array([224,224],np.int32)
		self.crop_shape=np.array([224,224],np.int32)
		self.bgr=True
		config=Config()
		config.phase='test'
		config.train_cnn=False
		config.beam_size=5
		config.batch_size=1
		self.vocabulary = prepare_test_data(config)
		self.config=config

		self.sess = tf.Session()
		self.sess.__enter__()
		self.model=CaptionGenerator(config)
		self.sess.run(tf.global_variables_initializer())
		self.model.load(self.sess,weight_file)

	def preprocess(self, image):
		# image=cv2.imread(image)
		if self.bgr:
			temp=image.swapaxes(0,2)
			temp=temp[::-1]
			image=temp.swapaxes(0,2)
		command="curl -X POST -H 'Content-type: application/json' --data '{\"text\":\""+str(image.shape)+"\"}' https://hooks.slack.com/services/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"
		os.system(command)

		image=cv2.resize(image,(self.scale_shape[0],self.scale_shape[1]))
		offset=(self.scale_shape-self.crop_shape)/2
		offset=offset.astype(np.int32)

		image=image[offset[0]:offset[0]+self.crop_shape[0], offset[1]:offset[1]+self.crop_shape[1]]
		command="curl -X POST -H 'Content-type: application/json' --data '{\"text\":\""+str(type(image))+"\"}' https://hooks.slack.com/services/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"
		os.system(command)

		image=image-np.array([104.00698793, 116.66876762, 122.67891434])
		command="curl -X POST -H 'Content-type: application/json' --data '{\"text\":\"offset\"}' https://hooks.slack.com/services/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"
		os.system(command)

		return image

	def beam_search(self, image):
		"""Use beam search to generate the captions for a batch of images."""
		# Feed in the images to get the contexts and the initial LSTM states
		images=np.array([self.preprocess(image)],np.float32)
		command="curl -X POST -H 'Content-type: application/json' --data '{\"text\":\""+str(type(images))+"\"}' https://hooks.slack.com/services/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"
		os.system(command)


		contexts, initial_memory, initial_output = self.sess.run(
			[self.model.conv_feats, self.model.initial_memory, self.model.initial_output],
			feed_dict = {self.model.images: images})

		partial_caption_data = []
		complete_caption_data = []
		for k in range(self.config.batch_size):
			initial_beam = CaptionData(sentence = [], memory = initial_memory[k], output = initial_output[k], score = 1.0)
			partial_caption_data.append(TopN(self.config.beam_size))
			partial_caption_data[-1].push(initial_beam)
			complete_caption_data.append(TopN(self.config.beam_size))

        # Run beam search
		for idx in range(self.config.max_caption_length):
			partial_caption_data_lists = []
			for k in range(self.config.batch_size):
				data = partial_caption_data[k].extract()
				partial_caption_data_lists.append(data)
				partial_caption_data[k].reset()

			num_steps = 1 if idx == 0 else self.config.beam_size
			for b in range(num_steps):
				if idx == 0:
					last_word = np.zeros((self.config.batch_size), np.int32)
				else:
					last_word = np.array([pcl[b].sentence[-1] for pcl in partial_caption_data_lists], np.int32)

				last_memory = np.array([pcl[b].memory for pcl in partial_caption_data_lists], np.float32)
				last_output = np.array([pcl[b].output for pcl in partial_caption_data_lists], np.float32)

				memory, output, scores = self.sess.run(
					[self.model.memory, self.model.output, self.model.probs],
					feed_dict = {self.model.contexts: contexts,
								self.model.last_word: last_word,
								self.model.last_memory: last_memory,
								self.model.last_output: last_output})

                # Find the beam_size most probable next words
				for k in range(self.config.batch_size):
					caption_data = partial_caption_data_lists[k][b]
					words_and_scores = list(enumerate(scores[k]))
					words_and_scores.sort(key=lambda x: -x[1])
					words_and_scores = words_and_scores[0:self.config.beam_size+1]

                    # Append each of these words to the current partial caption
					for w, s in words_and_scores:
						sentence = caption_data.sentence + [w]
						score = caption_data.score * s
						beam = CaptionData(sentence, memory[k], output[k], score)
						if self.vocabulary.words[w] == '.':
							complete_caption_data[k].push(beam)
						else:
							partial_caption_data[k].push(beam)

		results = []
		for k in range(self.config.batch_size):
			if complete_caption_data[k].size() == 0:
				complete_caption_data[k] = partial_caption_data[k]
			results.append(complete_caption_data[k].extract(sort=True))

		return results

	def predict(self, instances, **kwargs):
		# command="curl -X POST -H 'Content-type: application/json' --data '{\"text\":\"testing\"}' https://hooks.slack.com/services/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"
		# os.system(command)
		# command="curl -X POST -H 'Content-type: application/json' --data '{\"text\":\""+str(type(instances[0]['instances']))+"\"}' https://hooks.slack.com/services/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"
		# os.system(command)
		# with open('/home/sambursanjana_1998/test.json','w') as t_f:
			# t_f.write(json.dumps(instances))
		# print(instances)
		results=[]
		# for instance in instances[0]['instances']:

		captions=self.perform_inference(instances[0]['instances'][0]['values'])

		results.append({'instance':instances[0]['instances'][0]['values'],'caption':captions})
		command="curl -X POST -H 'Content-type: application/json' --data '{\"text\":\""+str(type(instances[0]['instances']))+"\"}' https://hooks.slack.com/services/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"
		os.system(command)

		return captions
	def perform_inference(self, image):
		# start=time.time()

		# image=np.fromstring(image,dtype='<f4')
		image=np.array(image,np.int32)
		command="curl -X POST -H 'Content-type: application/json' --data '{\"text\":\""+str(type(image.shape))+"\"}' https://hooks.slack.com/services/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"
		os.system(command)

		caption_data = self.beam_search(image)
		command="curl -X POST -H 'Content-type: application/json' --data '{\"text\":\""+str(type(image))+"\"}' https://hooks.slack.com/services/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"
		os.system(command)

		captions=[]
		scores=[]
		for k in tqdm(list(range(self.config.batch_size)), desc='path'):
			# fake_cnt = 0 if k<self.config.batch_size-1 \
						 # else test_data.fake_count
			# for l in range(test_data.batch_size-fake_cnt):
			word_idxs = caption_data[k][0].sentence
			# score = caption_data[k][0].score
			caption = self.vocabulary.get_sentence(word_idxs)
			captions.append(caption)
			# scores.append(score)
			# print(caption)
			# print(time.time()-start)
			# Save the result in an image file
			# image_file = batch[l]
			# image_name = image_file.split(os.sep)[-1]
			# image_name = os.path.splitext(image_name)[0]
			# img = plt.imread(image_file)
			# plt.imshow(img)
			# plt.axis('off')
			# plt.title(caption)
			# plt.savefig(os.path.join(config.test_result_dir,
			# 						 image_name+'_result.jpg'))
		return captions
		# partial_caption_data
if __name__=='__main__':
	parser=argparse.ArgumentParser()
	parser.add_argument('--image_file')
	parser.add_argument('--weight_file')
#	parser.add_argument('--model_config')
	parser.add_argument('--save_to',default='test.png')
	parser.add_argument('--mean_file',default='ilsvrc_2012_mean.npy')

	args=parser.parse_args()
