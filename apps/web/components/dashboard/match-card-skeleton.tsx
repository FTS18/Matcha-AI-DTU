import React from "react";

export const MatchCardSkeleton = () => {
  return (
    <div className="card relative group bg-card/20 backdrop-blur-md border border-white/5 overflow-hidden animate-pulse">
      {/* 左侧装饰条占位 (桌面端) */}
      <div className="hidden lg:block absolute left-0 top-0 bottom-0 w-1 bg-white/5" />

      {/* ── 移动端布局 ── */}
      <div className="lg:hidden">
        {/* 上部缩略图占位 */}
        <div className="relative w-full h-40 bg-white/5 flex items-center justify-center">
          <div className="w-12 h-12 rounded-full bg-white/5" />
          {/* 左上角状态徽章占位 */}
          <div className="absolute top-3 left-3 w-20 h-5 bg-white/5 border border-white/5" />
          {/* 右上角时间占位 */}
          <div className="absolute top-3 right-3 w-14 h-4 bg-white/5" />
          {/* 左下角标题与 ID 占位 */}
          <div className="absolute bottom-3 left-3 space-y-2 w-2/3">
            <div className="h-4 bg-white/10 rounded-sm w-3/4" />
            <div className="h-2.5 bg-white/5 rounded-sm w-1/2" />
          </div>
        </div>

        {/* 下部数据统计区占位 */}
        <div className="flex items-center justify-around px-2 py-3 bg-black/10 border-t border-white/5">
          <div className="flex flex-col items-center gap-1.5 flex-1">
            <div className="w-8 h-2.5 bg-white/5 rounded-sm" />
            <div className="w-12 h-4 bg-white/5 rounded-sm" />
          </div>
          <div className="w-px h-6 bg-white/5" />
          <div className="flex flex-col items-center gap-1.5 flex-1">
            <div className="w-8 h-2.5 bg-white/5 rounded-sm" />
            <div className="w-6 h-4 bg-white/5 rounded-sm" />
          </div>
          <div className="w-px h-6 bg-white/5" />
          <div className="flex flex-col items-center gap-1.5 flex-1">
            <div className="w-8 h-2.5 bg-white/5 rounded-sm" />
            <div className="w-6 h-4 bg-white/5 rounded-sm" />
          </div>
        </div>
      </div>

      {/* ── 桌面端布局 ── */}
      <div className="hidden lg:flex lg:h-18 items-stretch relative overflow-hidden">
        {/* 左侧缩略图 */}
        <div className="w-30 h-full shrink-0 border-r border-white/10 bg-white/5" />
        
        {/* 中间信息区 */}
        <div className="flex-1 flex flex-col justify-center px-6 py-2 gap-2">
          <div className="flex items-center gap-3">
            <div className="w-14 h-4 bg-white/10 rounded-sm" />
            <div className="w-48 h-5 bg-white/10 rounded-sm" />
          </div>
          <div className="w-32 h-3 bg-white/5 rounded-sm" />
        </div>

        {/* 右侧统计与按钮 */}
        <div className="flex items-stretch border-l border-white/5">
          {/* Stats 占位 */}
          <div className="flex items-center gap-6 px-6">
            <div className="flex flex-col items-center gap-1">
              <div className="w-10 h-2 bg-white/5 rounded-sm" />
              <div className="w-8 h-3.5 bg-white/5 rounded-sm" />
            </div>
            <div className="flex flex-col items-center gap-1">
              <div className="w-10 h-2 bg-white/5 rounded-sm" />
              <div className="w-4 h-3.5 bg-white/5 rounded-sm" />
            </div>
            <div className="flex flex-col items-center gap-1">
              <div className="w-10 h-2 bg-white/5 rounded-sm" />
              <div className="w-4 h-3.5 bg-white/5 rounded-sm" />
            </div>
          </div>
          
          {/* Highlights 按钮 */}
          <div className="flex items-center justify-center w-30 border-l border-white/5 px-4">
            <div className="w-full h-7 bg-white/5 rounded-sm border border-white/5" />
          </div>

          {/* Actions 操作区 */}
          <div className="flex items-center justify-center w-30 border-l border-white/10 bg-white/2 px-4">
            <div className="w-8 h-8 rounded-full bg-white/5" />
          </div>
        </div>
      </div>
    </div>
  );
};
