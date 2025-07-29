#include <cstdint>
#include <algorithm>
#include <cstdio>
#include <vector>

using std::uint32_t;
using std::size_t;

#define NCCL_STEPS 8

template<typename X, typename Y, typename Z = decltype(X()+Y())>
constexpr Z divUp(X x, Y y) {
  return (x+y-1)/y;
}

static size_t alignUp(size_t Size, size_t Alignment) {
  return ((Size + Alignment - 1) & ~(Alignment - 1));
}

struct CBD {
  uint32_t countLo;
  uint32_t countMid;
  uint32_t countHi;
  uint32_t chunkGrainsLo;
  uint32_t chunkGrainsMid;
  uint32_t chunkGrainsHi;
};

struct ncclDevWorkColl {
  uint32_t channelLo;
  uint32_t channelHi;
  CBD cbd;
};

template<typename Int>
void ncclCollCbdPart(
    struct ncclDevWorkColl* work, uint32_t channelId, int eltSize,
    Int* partOffset, Int* partCount, Int* chunkCount
  ) {
  int eltPerGrain = 512/eltSize;
  int nMidChannels = work->channelHi - work->channelLo - 1;
  // We can assum that nMidChannels<0 implies countMid==0, which let's us assume
  // that countMid*nMidChannels == 0.
  if (channelId == work->channelLo) {
    *partOffset = 0;
    *partCount = work->cbd.countLo;
    *chunkCount = work->cbd.chunkGrainsLo*eltPerGrain;
  } else if (channelId == work->channelHi) {
    *partOffset = work->cbd.countLo + nMidChannels*work->cbd.countMid;
    *partCount = work->cbd.countHi;
    *chunkCount = work->cbd.chunkGrainsHi*eltPerGrain;
  } else {
    int mid = channelId - work->channelLo - 1;
    *partOffset = work->cbd.countLo + mid*work->cbd.countMid;
    *partCount = work->cbd.countMid;
    *chunkCount = work->cbd.chunkGrainsMid*eltPerGrain;
  }
}

static void scheduleCollTasksToPlan(
    int nChannels, size_t datatype_size, size_t count, struct ncclDevWorkColl *devWork, size_t nccl_buff_size=1<<20
  ) {

  constexpr size_t MinTrafficPerChannel = 512;
  size_t trafficPerChannel = 0;
  int channelId = 0 ;
  size_t currentTraffic = 0;
  size_t elementSize = datatype_size;
  size_t trafficBytes = count*elementSize*2;
  size_t grainSize = 512;

  trafficPerChannel = std::max<size_t>(MinTrafficPerChannel, trafficBytes/nChannels);
  channelId = 0;
  currentTraffic = 0;
  constexpr size_t cellSize = 16;
  int elementsPerCell = cellSize/elementSize;
  size_t cells = divUp(count*elementSize, cellSize);
  int trafficPerByte = 2 ; 
  size_t trafficPerElement = elementSize*trafficPerByte;
  size_t trafficPerCell = cellSize*trafficPerByte;
  size_t cellsPerChannel = std::min<size_t>(cells, divUp(trafficPerChannel, trafficPerCell));
  size_t cellsLo;
  if (channelId+1 == nChannels) { // On last channel everything goes to "lo"
    cellsLo = cells;
  } else {
    cellsLo = std::min(cells, (trafficPerChannel-currentTraffic)/trafficPerCell);
  }
  int nMidChannels = (cells-cellsLo)/cellsPerChannel;
  size_t cellsHi = (cells-cellsLo)%cellsPerChannel;
  if (cellsHi == 0 && nMidChannels != 0) {
    cellsHi = cellsPerChannel;
    nMidChannels -= 1;
  }
  if (cellsLo == 0) { // Least channel skipped. Make the next channel the new least.
    channelId += 1;
    if (nMidChannels == 0) { cellsLo = cellsHi; cellsHi = 0; }
    else { cellsLo = cellsPerChannel; nMidChannels -= 1; }
  }
  size_t countMid = nMidChannels!=0 ? cellsPerChannel*elementsPerCell : 0;
  size_t countLo = cellsLo*elementsPerCell;
  size_t countHi = cellsHi*elementsPerCell;
  (countHi != 0 ? countHi : countLo) -= cells*elementsPerCell - count;

  nChannels = (countLo!=0 ? 1 : 0) + nMidChannels + (cellsHi!=0 ? 1 : 0);

  devWork->channelLo = channelId;
  devWork->channelHi = channelId + nChannels-1;
  devWork->cbd.countLo = countLo;
  devWork->cbd.countMid = countMid;
  devWork->cbd.countHi = countHi;

  /* came from calcCollChunking */
  uint32_t chunkSize = (nccl_buff_size/NCCL_STEPS) * (NCCL_STEPS/2) / grainSize * grainSize;
  uint32_t directFlags=0;
  if (countLo != 0) {
    devWork->cbd.chunkGrainsLo = chunkSize / grainSize;
  }
  if (countHi != 0) {
    devWork->cbd.chunkGrainsHi = chunkSize / grainSize;
  }
  if (nMidChannels != 0) {
    devWork->cbd.chunkGrainsMid = chunkSize / grainSize;
  }

  std::fprintf(stderr, "Collective %s(elemsize: %ld, Ring, Simple) count=%ld channel{Lo..Hi}={%d..%d} count{Lo,Mid,Hi}={%ld,%ld,%ld} chunkBytes{Lo,Mid,Hi}={%d,%d,%d}\n",
    "AllReduce", 
    (long)datatype_size, 
    (long)count, devWork->channelLo, devWork->channelHi,
    (long)devWork->cbd.countLo, (long)devWork->cbd.countMid, (long)devWork->cbd.countHi,
    int(devWork->cbd.chunkGrainsLo*grainSize),
    int(devWork->cbd.chunkGrainsMid*grainSize),
    int(devWork->cbd.chunkGrainsHi*grainSize));
}

struct ChunkInfo {
  int ringIx;
  int nranks; 
  size_t channelId;
  size_t count;
  size_t nChannel;
  size_t dtype_size;
  std::vector<int> tagChunks;
  std::vector<int> tagOffsets;
  std::vector<int> sliceSizes;
};


void fillChunkInfo(struct ChunkInfo* chunkInfo, int ringIx, int nranks, size_t channelId, size_t count, size_t nChannel, size_t dtype_size, size_t nccl_buff_size = 1 << 22) {
  ssize_t gridOffset;
  ssize_t channelCount; 
  ssize_t chunkCount;

  struct ncclDevWorkColl work;
  scheduleCollTasksToPlan(nChannel, dtype_size, count, &work);
  ncclCollCbdPart(&work, channelId, dtype_size, &gridOffset, &channelCount, &chunkCount);
  chunkInfo->ringIx = ringIx; 
  chunkInfo->nranks = nranks;
  chunkInfo->channelId = channelId;
  chunkInfo->count = count;
  chunkInfo->nChannel = nChannel;
  chunkInfo->dtype_size = dtype_size;

  std::fprintf(stderr, "chunkCount %ld\n", chunkCount);
  const ssize_t loopCount = nranks * chunkCount;
  ssize_t offset;
  int nelem;
  int chunk;
  int tagChunk;
  size_t tagOffset;
  int offset_genericOp = 0;
  int stepSize = nccl_buff_size / 8 / dtype_size; // comm->buffSizes[NCCL_PROTO_SIMPLE] / NCCL_STEP; 
  
  int slice = 0;

  for (ssize_t elemOffset = 0; elemOffset < channelCount; elemOffset += loopCount) {
    ssize_t remCount = channelCount - elemOffset;
    ssize_t chunkOffset;

    if (remCount < loopCount) chunkCount = alignUp(divUp(remCount, nranks), 16/dtype_size);

    auto modRanks = [&](int r)->int {
      return r - (r >= nranks ? nranks : 0);
    };

    // step k-1: reduce this buffer and data, which will produce the final
    // result that we store in this data and push to the next GPU
    chunk = ringIx + 0;
    chunkOffset = chunk * chunkCount;
    offset = gridOffset + elemOffset + chunkOffset;
    nelem = (int)std::min<size_t>(chunkCount, remCount - chunkOffset);
    tagChunk = (ringIx == 0 || ringIx == (nranks - 1)) ? static_cast<int>(offset) : -1;
    slice = 0;
    int sliceSize = stepSize*2; // 2: stepperslice
    sliceSize = std::max<size_t>(divUp(nelem, 16*2/*2: SlicePerChunk*/)*16, sliceSize/32);
    offset_genericOp = 0;
    do {
      sliceSize = sliceSize < nelem-offset_genericOp ? sliceSize : nelem-offset_genericOp;
      tagOffset = tagChunk != -1 ? ((tagChunk + offset_genericOp) * dtype_size) : -1;
      offset_genericOp += sliceSize;
      slice += 1;
      chunkInfo->tagOffsets.emplace_back(tagOffset);
      chunkInfo->tagChunks.emplace_back(tagChunk);
      chunkInfo->sliceSizes.emplace_back(sliceSize);
      std::fprintf(stderr, "tagChunk: %d, tagOffset = %ld, nelem=%d, size=%ld, ringIx %d\n", tagChunk, tagOffset, nelem, sliceSize*dtype_size, ringIx);
    } while (slice < 2 && offset_genericOp < nelem);


    // k-2 steps: copy to next GPU
    for (int j = 1; j < nranks - 1; ++j) {
      chunk = modRanks(ringIx + nranks - j);
      chunkOffset = chunk * chunkCount;
      offset = gridOffset + elemOffset + chunkOffset;
      nelem = (int)std::min(chunkCount, remCount - chunkOffset);
      tagChunk = (ringIx == (nranks - 1)) ? static_cast<int>(offset) : -1;
      slice = 0;
      int sliceSize = stepSize*2; // 2: stepperslice
      sliceSize = std::max<size_t>(divUp(nelem, 16*2/*2: SlicePerChunk*/)*16, sliceSize/32);
      offset_genericOp = 0;
      do {
        sliceSize = sliceSize < nelem-offset_genericOp ? sliceSize : nelem-offset_genericOp;
        tagOffset = tagChunk != -1 ? ((tagChunk + offset_genericOp) * dtype_size) : -1;
        offset_genericOp += sliceSize;
        slice += 1;
        chunkInfo->tagOffsets.emplace_back(tagOffset);
        chunkInfo->tagChunks.emplace_back(tagChunk);
        chunkInfo->sliceSizes.emplace_back(sliceSize);
        std::fprintf(stderr, "tagChunk: %d, tagOffset = %ld, nelem=%d, size=%ld, ringIx %d\n", tagChunk, tagOffset, nelem, sliceSize*dtype_size, ringIx);
      } while (slice < 2 && offset_genericOp < nelem);
    }
  } 
}

int main(int argc, char * argv[]) {
    int ringIx = atoi(argv[1]);
    const int nranks = atoi(argv[2]);
    size_t channelId = atoi(argv[3]);
    size_t count = atoi(argv[4]);
    size_t nChannel = atoi(argv[5]);
    size_t dtype_size = atoi(argv[6]);

    struct ChunkInfo chunkInfo;
    fillChunkInfo(&chunkInfo, ringIx, nranks, channelId, count, nChannel, dtype_size);

    return 0;
}