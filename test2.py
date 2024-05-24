import argparse
import numpy as np
import fixed_env as env
import load_trace
import time


S_INFO = 6  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_LEN = 8  # take how many frames in the past
A_DIM = 6
VIDEO_BIT_RATE = [300,750,1200,1850,2850,4300]  # Kbps
M_IN_K = 1000.0
# REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000000
RESEVOIR = 20  # BB
CUSHION = 8  # BB

parser = argparse.ArgumentParser(description='Buffer-based')
parser.add_argument('--lin', action='store_true', help='QoE_lin metric')
parser.add_argument('--log', action='store_true', help='QoE_log metric')
parser.add_argument('--FCC', action='store_true', help='Test in FCC dataset')
parser.add_argument('--HSDPA', action='store_true', help='Test in HSDPA dataset')
parser.add_argument('--Oboe', action='store_true', help='Test in Oboe dataset')

def main():
    args = parser.parse_args()
    if args.lin:
        qoe_metric = 'results_lin'
    elif args.log:
        qoe_metric = 'results_log'
    else:
        print('Please select the QoE Metric!')
    
    if args.FCC:
        dataset = 'fcc'
    elif args.HSDPA:
        dataset = 'HSDPA'
    elif args.Oboe:
        dataset = 'Oboe'
    else:
        print('Please select the dataset!')
    
    dataset_path = './traces_' + dataset + '/'
    Log_file_path = './' + qoe_metric + '/' + dataset + '/log_sim_test2'

    np.random.seed(RANDOM_SEED)

    assert len(VIDEO_BIT_RATE) == A_DIM

    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(dataset_path)

    net_env = env.Environment(all_cooked_time=all_cooked_time,
                              all_cooked_bw=all_cooked_bw)

    log_path = Log_file_path + '_' + all_file_names[net_env.trace_idx]

    log_file = open(log_path, 'w')
    

    epoch = 0
    time_stamp = 0

    last_bit_rate = DEFAULT_QUALITY
    bit_rate = DEFAULT_QUALITY

    r_batch = []

    video_count = 0
    bwe_vec = []
    queue_time = 23
    is_playing = 1
    levels = [{'bitrate': bitrate} for bitrate in VIDEO_BIT_RATE]
    max_level_imp = 0
    min_level_imp = 0
    horizon = 3
    queue_target = 10
    delta = 5
    q_h = queue_target + delta
    q_l = queue_target
    q_ll = q_l * 0.8
    k1 = 0.01
    k2 = 0.001
    last_level = 0
    cold_start = True
    prec_state = 0
    t_last = -1

    while True:  # serve video forever
        # the action is from the last decision
        # this is to make the framework similar to the real
        delay, sleep_time, buffer_size, rebuf, \
        video_chunk_size, next_video_chunk_sizes, \
        end_of_video, video_chunk_remain = \
            net_env.get_video_chunk(bit_rate)

        time_stamp += delay  # in ms
        time_stamp += sleep_time  # in ms

        # reward is video quality - rebuffer penalty
        if qoe_metric == 'results_lin':
            REBUF_PENALTY = 4.3
            reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
                    - REBUF_PENALTY * rebuf \
                    - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] -
                                            VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K
        else:
            REBUF_PENALTY = 2.66
            log_bit_rate = np.log(VIDEO_BIT_RATE[bit_rate] / float(VIDEO_BIT_RATE[0]))
            log_last_bit_rate = np.log(VIDEO_BIT_RATE[last_bit_rate] / float(VIDEO_BIT_RATE[0]))

            reward = log_bit_rate \
                    - REBUF_PENALTY * rebuf \
                    - SMOOTH_PENALTY * np.abs(log_bit_rate - log_last_bit_rate)

        last_bit_rate = bit_rate

        # log time_stamp, bit_rate, buffer_size, reward
        log_file.write(str(time_stamp / M_IN_K) + '\t' +
                       str(VIDEO_BIT_RATE[bit_rate]) + '\t' +
                       str(buffer_size) + '\t' +
                       str(rebuf) + '\t' +
                       str(video_chunk_size) + '\t' +
                       str(delay) + '\t' +
                       str(reward) + '\n')
        log_file.flush()

        # INIZIO ELASTIC
        
        max_bitrate = levels[max_level_imp]['bitrate']
        min_bitrate = levels[min_level_imp]['bitrate']

        if bit_rate >= 2 * max_bitrate:
            bwe_vec.append(2 * max_bitrate)
        else:
            bwe_vec.append(bit_rate)
        bwe_vec = bwe_vec[-horizon:]

        bwe_filt = len(bwe_vec) / sum(1.0 / v for v in bwe_vec)

        queue_time = abs(queue_time)
        e = 0

        if queue_time > q_h:
            zero_int_error = (prec_state == 0)
            prec_state = 2
            e = queue_time - q_h
        elif queue_time < q_l:
            zero_int_error = (prec_state == 2)
            prec_state = 0
            e = queue_time - q_l
        else:
            prec_state = 1
            zero_int_error = True

        d = 1 if is_playing else 0

        if t_last < 0:
            delta_time = 0
            int_error = e
        else:
            ts = time.time() * 1000
            delta_time = (ts - t_last) / 1000
            if zero_int_error:
                int_error = 0
            int_error += delta_time * e
        t_last = time.time() * 1000

        den = 1.0 - (k1 * e) - (k2 * int_error)

        if queue_time < q_l or queue_time > q_h:
            if queue_time < q_ll and cold_start:
                den = 1.2
                if zero_int_error:
                    int_error -= (delta_time * e)
            elif cold_start:
                cold_start = False

            if den <= 0 or (bwe_filt / den) >= max_bitrate:
                u = max_bitrate + 10
                if zero_int_error:
                    int_error -= (delta_time * e)
            elif (bwe_filt / den) <= min_bitrate:
                u = min_bitrate
                if zero_int_error:
                    int_error -= (delta_time * e)
            else:
                u = bwe_filt / den

            level = 0
            for i, l in enumerate(levels):
                if u >= l['bitrate']:
                    level = i
            level_u = level

            if queue_time > q_h and level_u < last_level:
                level_u = last_level
            else:
                last_level = level_u
        else:
            level_u = last_level
        
        bit_rate = int(bit_rate)

        # FINE ELASTIC

        if end_of_video:
            log_file.write('\n')
            log_file.close()

            last_bit_rate = DEFAULT_QUALITY
            bit_rate = DEFAULT_QUALITY  # use the default action here
            r_batch = []

            print("video count", video_count)
            video_count += 1

            if video_count > len(all_file_names):
                break

            log_path = Log_file_path + '_' + all_file_names[net_env.trace_idx]
            log_file = open(log_path, 'w')


if __name__ == '__main__':
    main()