struct cuda_future
{
  cuda_future(cudaStream_t stream, cudaEvent_t event)
    : stream_(stream), event_(event)
  {}

  ~cuda_future()
  {
    cudaEventDestroy(event_);
  }

  void wait()
  {
    cudaEventSynchronize(event_);
  }

  cudaStream_t stream_;
  cudaEvent_t event_;
};

inline cuda_future make_ready_cuda_future(cudaStream_t stream)
{
  cudaEvent_t event;
  cudaEventCreateWithFlags(&event, cudaEventDisableTiming);

  return {stream, event};
}


template<class F>
__global__ void kernel(F f)
{
  f();
}

struct simple_cuda_then_executor
{
  template<class Function>
  cuda_future then_execute(Function f, cuda_future& fut) const
  {
    // before invoking f, make the stream wait on the previous event (stored in fut)
    cudaStreamWaitEvent(fut.stream_, fut.event_, 0);
    kernel<<<1,1,0,fut.stream_>>>(f);

    // record a new event corresponding to f's invocation
    cudaEvent_t new_event;
    cudaEventCreateWithFlags(&new_event, cudaEventDisableTiming);
    cudaEventRecord(new_event, fut.stream_);

    // return a new cuda_future corresponding to the completion of f
    return {fut.stream_, new_event};
  }
};

