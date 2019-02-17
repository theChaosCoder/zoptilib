#
# Zoptilib - helper script for the Zopti optimizer
#
# -supported similarity metrics are 'ssim', 'gmsd' and 'vmaf' (note: currently cannot have both gmsd and vmaf at the same time)
# -measures the runtime in milliseconds when 'time' is added to metrics list
# 
# Requirements: muvsfunc (for SSIM and GMSD), VapourSynth-VMAF (for VMAF)
# 
# Usage examples:
#
#   from zoptilib import Zopti
#   output_file = r'results.txt' # output out1="SSIM: MAX(float)" out2="time: MIN(time) ms" file="results.txt"
#   zopti = Zopti(output_file, metrics=['ssim', 'time'])	# initialize output file and chosen metrics 
#															# make sure metrics match what is defined at the line above
#															# Zopti starts measuring runtime at this point so call it just before the processing you want to measure
#
#   # define parameters to optimize (in the comments)
#   super_pel = 4					# optimize super_pel = _n_ | 2,4 | super_pel
#   super_sharp = 2					# optimize super_sharp = _n_ | 0..2 | super_sharp
#   super_rfilter = 2				# optimize super_rfilter = _n_ | 0..4 | super_rfilter
#   super = core.mv.Super(orig, pel=super_pel, sharp=super_sharp, rfilter=super_rfilter)
#
#   ... process the video ...
#
#   zopti.run(orig, alternate)								# measure similarity of original and alternate videos, save results to output file
#															# note: the first clip should be the reference / original clip
#   														# the output video is the second clip when vmaf is not used and the first clip when vmaf is used
#
#         OR 
#
#   zopti = Zopti(output_file)								# initialize output file
#   zopti.addMetrics(['ssim','time'])						# add chosen metrics
#   ... process the video and measure similarity ...
#
#
# Changelog:
#  2019-01-23   v1.0   initial version
#

import vapoursynth as vs
import time
import muvsfunc as muv
import functools

core = vs.core

class FrameData:

	def __init__(self, name):
		self.name = name
		self.per_frame_data = {}
	
class Zopti:
		
	def __init__(self, output_file, metrics = None):

		self.valid_metrics = ['time', 'ssim', 'gmsd', 'mdsi', 'butteraugli', 'vmaf']
		self.supported_with_vmaf = ['time', 'ssim']				# these metrics can be read from the vmaf output	
		self.output_file = output_file
		self.vmaf_model = 0										# default model: vmaf_v0.6.1.pkl
		self.params = {
			"ssim": {},
			"gmsd": {},
			"mdsi": {},			
		}
	
		# measure total runtime
		self.start_time = time.perf_counter()
	
		# used metrics 
		self.metrics = []
		if metrics:
			self.addMetrics(metrics)				
		
		
	def addMetric(self, metric):
		if metric in self.valid_metrics:
			if metric == 'vmaf':
				if (len(set(self.metrics) & set(self.supported_with_vmaf)) != len(self.metrics)):
					raise NameError('Only these metrics are supported with vmaf: '+str(list(self.supported_with_vmaf)))
			elif metric not in self.supported_with_vmaf:
				if 'vmaf' in self.metrics:
					raise NameError('Only these metrics are supported with vmaf: '+str(list(self.supported_with_vmaf)))
			self.metrics.append(metric)
		else:
			raise NameError('Unknown metric "'+metric+'"')

	def addMetrics(self, metrics):
		for metric in metrics:
			self.addMetric(metric)
					
	def setVMAFModel(self, model):
		self.vmaf_model = model
	
	def addParams(self, metric, params):
		if metric in self.valid_metrics:
			if type(params) is dict:
				self.params[metric].update(params)
			else:
				raise NameError('The parameters list should look like this: dict(myVar=True, myOtherVar=0.4)')
		else:
			raise NameError('Unknown metric "'+metric+'"')
					
	def run(self, clip, alt_clip):
		
		if len(self.metrics) == 0:
			raise ValueError('No metrics defined')
	
		if 'vmaf' not in self.metrics:
	
			total_frames = clip.num_frames
			metricsSet = set(self.metrics)
			data = []
			prop_src = []
			for metric in self.metrics:
				filter_args = self.params[metric] if metric in self.params else {}
				if metric == 'gmsd':
					# calculate GMSD between original and alternate version
					alt_clip = muv.GMSD(alt_clip, clip, **filter_args)
					prop_src = [alt_clip]
					data.append(FrameData('gmsd'))
				elif metric == 'ssim':
					# calculate SSIM between original and alternate version
					alt_clip = muv.SSIM(alt_clip, clip, **filter_args)
					prop_src = [alt_clip]
					data.append(FrameData('ssim'))
				elif metric == 'mdsi':
					# calculate MDSI between original and alternate version
					alt_clip = muv.MDSI(alt_clip, clip, **filter_args)
					prop_src = [alt_clip]
					data.append(FrameData('mdsi'))
				elif metric == 'butteraugli':
					# calculate butteraugli between original and alternate version
					alt_clip = core.Butteraugli.butteraugli(alt_clip, clip)
					prop_src = [alt_clip]
					data.append(FrameData('butteraugli'))
				elif metric == 'time':
					data.append(FrameData('time'))
				else:
					raise NameError('Unknown metric '+metric)
			
			def save_per_frame_data(n, frame_data, f):
				prop_name = ''
				if (frame_data.name == 'gmsd'):
					prop_name = 'PlaneGMSD'
				elif (frame_data.name == 'ssim'):
					prop_name = 'PlaneSSIM'
				elif (frame_data.name == 'mdsi'):
					prop_name = 'FrameMDSI'
				elif (frame_data.name == 'butteraugli'):
					prop_name = '_Diff'
				elif (frame_data.name == 'time'):
					pass
				else:
					raise NameError('Unknown per frame type '+frame_data.name)

				if prop_name != '':
					frame_data.per_frame_data[n] = f.props[prop_name]
				
			
			# write per frame GMSD and/or SSIM and total runtime to a file
			def calc(n, f, clip, data):
				
				for frame_data in data:
					save_per_frame_data(n, frame_data, f)
						
				# when all frames have been processed write results to a file
				if (len(data[0].per_frame_data) == total_frames):
					runtime = time.perf_counter() - self.start_time
					with open(self.output_file, 'w') as file:
						for frame in data[0].per_frame_data.items():
							frame_nbr = frame[0]
							file.write(str(frame_nbr) + '; ')
							for frame_data in data:
								if frame_data.name == 'time':
									file.write('0.0; ')	
								else:							
									info = frame_data.per_frame_data[frame_nbr]
									file.write(str(info)+'; ')	
							file.write('\n')
							
						# write sum of per frame values and/or total runtime to last line
						file.write('stop ')
						for frame_data in data:
							if frame_data.name == 'time':
								file.write(str(runtime*1000)+' ')
							else:
								file.write(str(sum(frame_data.per_frame_data.values()))+' ')					
				
				return clip

			final = alt_clip.std.FrameEval(functools.partial(calc, clip=alt_clip, data=data), prop_src=prop_src)
			final.set_output()
			return final
		else:
			
			# VMAF
			# model: 	0 = vmaf_v0.6.1.pkl, 1 = vmaf_4k_v0.6.1.pkl
			# log_fmt: 	0 = xml, 1 = json
			# pool: 	0 = mean, 1 = harmonic mean, 2 = min
			# ci: 		return confidence interval True/False
			calc_ssim = 'ssim' in self.metrics
			final = core.vmaf.VMAF(clip, alt_clip, model=self.vmaf_model, log_path=self.output_file, log_fmt=0, ssim=calc_ssim, ms_ssim=False, pool=0, ci=False)		
			final.set_output()
			return final
			