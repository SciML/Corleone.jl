<template>
  <div class="gallery-container">
    <div class="controls">
      <div class="search-header">
        <input 
          v-model="searchQuery" 
          type="text" 
          placeholder="Search tutorials..." 
          class="search-input"
        />
        <button v-if="hasActiveFilters" @click="resetFilters" class="reset-btn">
          Reset All
        </button>
      </div>
      
      <div class="filter-bar">
        <button 
          v-for="tag in allTags" 
          :key="tag" 
          class="filter-chip"
          :class="{ active: selectedTags.has(tag) }"
          @click="toggleTag(tag)"
        >
          {{ tag }}
          <span v-if="selectedTags.has(tag)" class="check">✓</span>
        </button>
      </div>
    </div>

    <div class="grid">
      <div v-for="item in filteredData" :key="item.link" class="card">
        <div class="card-header">
          <span class="icon">{{ item.icon || '📄' }}</span>
          <h3>{{ item.title }}</h3>
        </div>

        <div class="tag-list">
          <span 
            v-for="tag in item.tags" 
            :key="tag" 
            class="tag-pill"
            :class="{ highlighted: selectedTags.has(tag) }"
          >
            {{ tag }}
          </span>
        </div>

        <p class="description">{{ item.description }}</p>
        
        <div class="card-footer">
          <a :href="item.link" class="link">Read Tutorial →</a>
        </div>
      </div>
    </div>

    <div v-if="filteredData.length === 0" class="empty-state">
      <p>No tutorials match your current filters.</p>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'

interface Tutorial {
  title: string;
  description: string;
  link: string;
  icon?: string;
  tags?: string[];
}

const props = defineProps<{ items: Tutorial[] }>()

const searchQuery = ref('')
const selectedTags = ref<Set<string>>(new Set())

// Extract unique tags
const allTags = computed(() => {
  const tags = new Set<string>()
  props.items.forEach(item => item.tags?.forEach(t => tags.add(t)))
  return Array.from(tags).sort()
})

const hasActiveFilters = computed(() => {
  return searchQuery.value !== '' || selectedTags.value.size > 0
})

function toggleTag(tag: string) {
  if (selectedTags.value.has(tag)) {
    selectedTags.value.delete(tag)
  } else {
    selectedTags.value.add(tag)
  }
}

function resetFilters() {
  searchQuery.value = ''
  selectedTags.value.clear()
}

const filteredData = computed(() => {
  return props.items.filter(item => {
    // Search filter
    const matchesSearch = item.title.toLowerCase().includes(searchQuery.value.toLowerCase()) ||
                          item.description.toLowerCase().includes(searchQuery.value.toLowerCase())
    
    // Multi-tag filter: Item must contain ALL selected tags
    const matchesTags = Array.from(selectedTags.value).every(t => item.tags?.includes(t))
    
    return matchesSearch && matchesTags
  })
})
</script>

<style scoped>
.controls {
  margin-bottom: 2rem;
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.search-header {
  display: flex;
  gap: 10px;
}

.search-input {
  flex-grow: 1;
  padding: 12px 16px;
  border-radius: 8px;
  border: 1px solid var(--vp-c-divider);
  background: var(--vp-c-bg-alt);
}

.reset-btn {
  font-size: 0.8rem;
  color: var(--vp-c-brand-1);
  cursor: pointer;
  font-weight: 600;
}

.filter-bar {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.filter-chip {
  padding: 4px 12px;
  border-radius: 20px;
  font-size: 0.85rem;
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-divider);
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 4px;
}

.filter-chip.active {
  background: var(--vp-c-brand-soft);
  color: var(--vp-c-brand-1);
  border-color: var(--vp-c-brand-1);
}

.check { font-size: 0.7rem; }

.grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 1.5rem;
}

.card {
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-divider);
  border-radius: 12px;
  padding: 1.5rem;
  display: flex;
  flex-direction: column;
}

.tag-pill {
  font-size: 0.7rem;
  background: var(--vp-c-default-soft);
  padding: 2px 8px;
  border-radius: 4px;
  margin-right: 5px;
  color: var(--vp-c-text-2);
}

/* Highlight tags in the card if they are part of the active filter */
.tag-pill.highlighted {
  background: var(--vp-c-brand-soft);
  color: var(--vp-c-brand-1);
  font-weight: bold;
}

.empty-state {
  text-align: center;
  padding: 3rem;
  color: var(--vp-c-text-2);
  border: 2px dashed var(--vp-c-divider);
  border-radius: 12px;
}

.link {
  font-weight: bold;
  color: var(--vp-c-brand-1);
  text-decoration: none;
}
</style>