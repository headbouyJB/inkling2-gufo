// Minimal GGUF v3 tensor-offset reader (mmap). MIT.
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <functional>
#include <string>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
struct TInfo { std::string name; int type = 0, nd = 0; uint64_t ne[4] = {1, 1, 1, 1}; uint64_t off = 0; size_t bytes = 0; };
struct Gguf {
  const uint8_t* base = nullptr; size_t size = 0; uint64_t data_off = 0; std::map<std::string, TInfo> t;
  const uint8_t* ptr(const std::string& n) const { return base + data_off + t.at(n).off; }
  bool has(const std::string& n) const { return t.count(n) > 0; }
  static size_t type_bytes(int ty, uint64_t n) {   // ggml types used here
    switch (ty) { case 0: return n * 4; case 1: return n * 2; case 8: return n / 32 * 34; case 40: return n / 64 * 36; case 30: return n * 2;
      default: return 0; }
  }
  bool open(const char* path) {
    int fd = ::open(path, O_RDONLY); if (fd < 0) return false;
    struct stat st; fstat(fd, &st); size = st.st_size;
    base = (const uint8_t*)mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0); if (base == MAP_FAILED) return false;
    const uint8_t* p = base;
    auto rd = [&](auto& v) { memcpy(&v, p, sizeof(v)); p += sizeof(v); };
    auto rstr = [&]() { uint64_t n; rd(n); std::string s((const char*)p, n); p += n; return s; };
    uint32_t magic, ver; rd(magic); rd(ver); if (magic != 0x46554747 || ver < 2) return false;
    uint64_t nt, nk; rd(nt); rd(nk);
    uint32_t align = 32;
    std::function<void(uint32_t)> skip;
    auto tsz = [](uint32_t ty) -> int { switch (ty) { case 0: case 1: case 7: return 1; case 2: case 3: return 2; case 4: case 5: case 6: return 4; default: return 8; } };
    skip = [&](uint32_t ty) {
      if (ty == 8) { rstr(); }
      else if (ty == 9) { uint32_t et; uint64_t n; rd(et); rd(n); for (uint64_t i = 0; i < n; i++) skip(et); }
      else p += tsz(ty);
    };
    for (uint64_t i = 0; i < nk; i++) {
      std::string k = rstr(); uint32_t ty; rd(ty);
      if (k == "general.alignment" && ty == 4) { rd(align); } else skip(ty);
    }
    for (uint64_t i = 0; i < nt; i++) {
      TInfo ti; ti.name = rstr(); uint32_t nd; rd(nd); ti.nd = nd; uint64_t n = 1;
      for (uint32_t d = 0; d < nd; d++) { rd(ti.ne[d]); n *= ti.ne[d]; }
      uint32_t ty; rd(ty); ti.type = ty; rd(ti.off); ti.bytes = type_bytes(ty, n); t[ti.name] = ti;
    }
    uint64_t pos = p - base; data_off = (pos + align - 1) / align * align;
    return true;
  }
};
